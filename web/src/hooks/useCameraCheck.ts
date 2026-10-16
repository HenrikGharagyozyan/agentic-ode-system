// The explicit camera check: runs ONLY when check() is called (a button click), never retries by itself.

import { useEffect, useRef, useState } from "react";

import { ApiError, checkCamera, isAbort } from "../api";
import type { CameraCheck } from "../types";

export type CheckState =
  | { kind: "idle" }
  | { kind: "running"; startedAt: number }
  | { kind: "done"; check: CameraCheck } // the check ran; check.ok tells whether the camera answered
  | { kind: "busy"; message: string } // 409: an experiment or another check owns the camera
  | { kind: "failed"; error: ApiError }; // offline, server error or unexpected answer

export function useCameraCheck(onFinished: () => void) {
  const [state, setState] = useState<CheckState>({ kind: "idle" });
  const ctrl = useRef<AbortController | null>(null);

  useEffect(() => () => ctrl.current?.abort(), []); // unmount: drop the request

  const check = async () => {
    if (ctrl.current) return; // one check at a time from this page
    ctrl.current = new AbortController();
    setState({ kind: "running", startedAt: Date.now() });
    try {
      setState({ kind: "done", check: await checkCamera(ctrl.current.signal) });
    } catch (e) {
      if (isAbort(e)) return;
      if (e instanceof ApiError && e.status === 409) setState({ kind: "busy", message: e.message });
      else setState({ kind: "failed", error: e instanceof ApiError ? e : new ApiError("invalid", String(e)) });
    } finally {
      ctrl.current = null;
      onFinished(); // refresh the status (busy flag, last_check)
    }
  };

  return { state, check };
}
