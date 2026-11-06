/**
 * Repro: clicking "Scroll to bottom" makes the button flicker.
 *
 * `scrollToBottomNow()` hides the button optimistically (`setAtBottom(true)`),
 * then the smooth navigation emits a scroll event for every animation frame.
 * While the camera is still travelling, `observeScroll()` reports
 * `owner === "navigation"` with `near === false`, so `logicallyAtBottom` flips
 * back to false and the button remounts (with its `animate-in` entrance) before
 * hiding again on arrival.
 *
 * A manual frame clock is required: the camera animates against
 * `performance.now()` over 220 ms, so on a loaded machine a real rAF tick can
 * finish the whole trip in one frame and the intermediate frames — the ones that
 * cause the flicker — never get sampled.
 */
import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ThreadViewport } from "@/components/thread/ThreadViewport";
import type { UIMessage } from "@/lib/types";

afterEach(() => {
  vi.restoreAllMocks();
});

/** Deterministic rAF + clock: one queued callback per 16 ms step. */
function installFrameClock() {
  let now = 0;
  let nextId = 1;
  const frames = new Map<number, FrameRequestCallback>();
  vi.spyOn(window, "requestAnimationFrame").mockImplementation((callback) => {
    const id = nextId;
    nextId += 1;
    frames.set(id, callback);
    return id;
  });
  vi.spyOn(window, "cancelAnimationFrame").mockImplementation((id) => {
    frames.delete(id);
  });
  vi.spyOn(performance, "now").mockImplementation(() => now);
  const step = async (): Promise<void> => {
    now += 16;
    const batch = [...frames.entries()];
    frames.clear();
    await act(async () => {
      for (const [, callback] of batch) callback(now);
    });
  };
  return { step, pendingFrames: () => frames.size };
}

function getScroller(container: HTMLElement): HTMLElement {
  const el = container.querySelector<HTMLElement>(".thread-viewport-scrollbar");
  if (!el) throw new Error("no scrollport");
  return el;
}

function setGeometry(el: HTMLElement, height: number, viewport: number, top: number) {
  Object.defineProperties(el, {
    scrollHeight: { configurable: true, value: height },
    clientHeight: { configurable: true, value: viewport },
    scrollTop: { configurable: true, writable: true, value: top },
  });
}

const messages: UIMessage[] = [
  { id: "u1", role: "user", content: "q1", turnId: "turn-1", createdAt: 1 },
  { id: "a1", role: "assistant", content: "a1", turnId: "turn-1", createdAt: 2 },
  { id: "u2", role: "user", content: "q2", turnId: "turn-2", createdAt: 3 },
  { id: "a2", role: "assistant", content: "a2", turnId: "turn-2", createdAt: 4 },
];

function buttonVisible(): boolean {
  return screen.queryByRole("button", { name: "Scroll to bottom" }) !== null;
}

describe("scroll-to-bottom button", () => {
  it("stays hidden for the whole trip once clicked", async () => {
    const clock = installFrameClock();
    const { container } = render(
      <ThreadViewport messages={messages} isStreaming={false} composer={<div>composer</div>} />,
    );
    const el = getScroller(container);
    setGeometry(el, 2400, 500, 1900); // max 1900: parked at the tail
    await clock.step();

    // Take the camera, then move far from the tail so the button appears.
    act(() => {
      fireEvent.wheel(el, { deltaY: -100 });
    });
    el.scrollTop = 400;
    act(() => {
      el.dispatchEvent(new Event("scroll"));
    });
    await clock.step();
    expect(buttonVisible()).toBe(true);

    // Click: this starts the smooth navigation and hides the button.
    fireEvent.click(screen.getByRole("button", { name: "Scroll to bottom" }));
    const samples: Array<{ top: number; visible: boolean }> = [{ top: el.scrollTop, visible: buttonVisible() }];

    // Emulate a browser: each animation frame writes scrollTop and emits scroll.
    for (let frame = 0; frame < 30; frame += 1) {
      // eslint-disable-next-line no-await-in-loop
      await clock.step();
      act(() => {
        el.dispatchEvent(new Event("scroll"));
      });
      samples.push({ top: Math.round(el.scrollTop), visible: buttonVisible() });
      if (Math.abs(el.scrollTop - 1900) < 1 && clock.pendingFrames() === 0) break;
    }

    const intermediate = samples.filter((s) => s.top > 400 && s.top < 1900);
    const reappeared = samples.filter((s) => s.visible);
    console.log("samples:", JSON.stringify(samples));
    // The trip must actually have been sampled frame by frame, or the assertion
    // below would pass vacuously.
    expect(intermediate.length).toBeGreaterThan(1);
    expect(reappeared.length).toBe(0);
    expect(Math.round(el.scrollTop)).toBe(1900);
  });
});
