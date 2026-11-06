/**
 * Scroll intent that arrives without a `wheel` event.
 *
 * Browser autoscroll (middle click) and scrollbar drags move the viewport and
 * emit only `scroll`. `observeScroll()` deliberately reads them as geometry, so
 * an automatic mode reclaims the camera on the next frame and pins the tail
 * again — the viewport visibly fights the user. These cases keep the wheel
 * behaviour as the reference and add the layout-churn case that must NOT be
 * mistaken for intent.
 */
import { act, fireEvent, render } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ThreadViewport } from "@/components/thread/ThreadViewport";
import { ThreadCameraController } from "@/components/thread/thread-camera";
import type { UIMessage } from "@/lib/types";

afterEach(() => {
  vi.restoreAllMocks();
});

async function flushAnimationFrame(): Promise<void> {
  await act(async () => {
    await new Promise<void>((resolve) => window.requestAnimationFrame(() => resolve()));
  });
}

function getScroller(container: HTMLElement): HTMLElement {
  const scroller = container.querySelector<HTMLElement>(".thread-viewport-scrollbar");
  if (!scroller) throw new Error("thread scrollport not found");
  return scroller;
}

function mockGeometry(scroller: HTMLElement, scrollHeight: number, clientHeight: number, scrollTop: number) {
  Object.defineProperties(scroller, {
    scrollHeight: { configurable: true, value: scrollHeight },
    clientHeight: { configurable: true, value: clientHeight },
    scrollTop: { configurable: true, writable: true, value: scrollTop },
  });
}

const threaded: UIMessage[] = [
  { id: "u1", role: "user", content: "old question", turnId: "turn-1", createdAt: 1 },
  { id: "a1", role: "assistant", content: "old answer", turnId: "turn-1", createdAt: 2 },
  { id: "u2", role: "user", content: "new question", turnId: "turn-2", createdAt: 3 },
];

function answer(isStreaming: boolean): UIMessage {
  return {
    id: "a2", role: "assistant", content: "final answer",
    turnId: "turn-2", isStreaming, createdAt: 4,
  };
}

/** One step of the browser's own scrolling: the scroller moved, `scroll` fires. */
function scrollUp(scroller: HTMLElement, top: number): void {
  scroller.scrollTop = top;
  act(() => {
    scroller.dispatchEvent(new Event("scroll"));
  });
}

/** A completed turn parked at the live tail: camera mode `follow-completion`. */
async function mountCompletedTurnAtTail(followTo = vi.spyOn(ThreadCameraController.prototype, "followTo").mockReturnValue("started")) {
  const { container, rerender } = render(
    <ThreadViewport
      messages={threaded}
      isStreaming
      composer={<div>composer</div>}
      activeTurnId="turn-2"
      activeTurnStartedHere
    />,
  );
  const scroller = getScroller(container);
  mockGeometry(scroller, 1904, 500, 1404);
  Object.defineProperty(container.querySelector<HTMLElement>('[data-user-prompt-id="u2"]'), "offsetTop", {
    configurable: true,
    value: 1420,
  });

  await act(async () => {
    rerender(
      <ThreadViewport
        messages={[...threaded, answer(true)]}
        isStreaming
        composer={<div>composer</div>}
        activeTurnId="turn-2"
      />,
    );
  });
  mockGeometry(scroller, 1932, 500, 1432);
  await act(async () => {
    rerender(
      <ThreadViewport messages={[...threaded, answer(false)]} isStreaming={false} composer={<div>composer</div>} />,
    );
  });
  await flushAnimationFrame();
  // Seed the previous frame at the tail, exactly as a real browser does.
  act(() => {
    scroller.dispatchEvent(new Event("scroll"));
  });
  await flushAnimationFrame();
  followTo.mockClear();
  return { scroller, followTo };
}

describe("scroll intent without a wheel event", () => {
  it("keeps the reading position when a wheel event starts the gesture", async () => {
    const { scroller, followTo } = await mountCompletedTurnAtTail();
    act(() => {
      fireEvent.wheel(scroller, { deltaY: -100 });
    });
    scrollUp(scroller, 100);
    await flushAnimationFrame();
    expect(followTo).not.toHaveBeenCalled();
    expect(scroller.scrollTop).toBe(100);
  });

  it("keeps the reading position when autoscroll moves the viewport with no input event", async () => {
    const { scroller, followTo } = await mountCompletedTurnAtTail();
    // Middle click arms autoscroll; the engine then emits only `scroll`.
    act(() => {
      fireEvent.pointerDown(scroller, { button: 1 });
    });
    scrollUp(scroller, 100);
    await flushAnimationFrame();
    expect(followTo).not.toHaveBeenCalled();
    expect(scroller.scrollTop).toBe(100);
  });

  it("keeps the reading position for an unannounced upward scroll in a restored thread", async () => {
    const navigateTo = vi.spyOn(ThreadCameraController.prototype, "navigateTo").mockReturnValue("settled");
    const jumpTo = vi.spyOn(ThreadCameraController.prototype, "jumpTo");
    const { container } = render(
      <ThreadViewport
        messages={threaded}
        isStreaming={false}
        composer={<div>composer</div>}
        scrollToBottomSignal={1}
      />,
    );
    const scroller = getScroller(container);
    mockGeometry(scroller, 1932, 500, 1432);
    await flushAnimationFrame();
    act(() => {
      scroller.dispatchEvent(new Event("scroll"));
    });
    jumpTo.mockClear();
    navigateTo.mockClear();

    scrollUp(scroller, 100);
    await flushAnimationFrame();
    // `follow-latest` reasserts the boundary with a hard jump, which is the more
    // visible variant of the same fight.
    expect(jumpTo).not.toHaveBeenCalled();
    expect(scroller.scrollTop).toBe(100);
  });

  it("still treats a layout shrink as automatic, not as user intent", async () => {
    const { scroller, followTo } = await mountCompletedTurnAtTail();
    // Content above the viewport was removed: scroll anchoring lowers both the
    // offset and the boundary by the same travel, so this is not a user move.
    mockGeometry(scroller, 2168, 500, 1200);
    scroller.scrollTop = 1200;
    act(() => {
      scroller.dispatchEvent(new Event("scroll"));
    });
    await flushAnimationFrame();
    expect(followTo).toHaveBeenCalledTimes(1);
    expect(followTo).toHaveBeenLastCalledWith(1668);
  });
});
