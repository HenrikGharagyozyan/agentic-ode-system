// Polls the two SAFE read endpoints (health, camera status) every few seconds. Never the camera check.
// One request round at a time: the next round is scheduled only after the last one finished.

import { useEffect, useRef, useState } from "react";

import { ApiError, getCameraStatus, getHealth, isAbort } from "../api";
import type { CameraStatus, Health } from "../types";

export const POLL_MS = 4000;

export interface BackendStatus {
  health: Health | null; // null: not known yet, or the backend is offline
  camera: CameraStatus | null;
  error: ApiError | null; // the error of the last round, or null
  loading: boolean; // true until the first round finished
  updatedAt: Date | null; // when the last round finished successfully
}

export function useBackendStatus(): BackendStatus & { refresh: () => void } {
  const [status, setStatus] = useState<BackendStatus>({
    health: null,
    camera: null,
    error: null,
    loading: true,
    updatedAt: null,
  });
  const refreshRef = useRef<() => void>(() => {});

  useEffect(() => {
    // everything below belongs to this one mount; the cleanup stops it completely
    const ctrl = new AbortController();
    let busy = false;
    let stopped = false;
    let timer: number | undefined;

    const round = async () => {
      if (busy) return; // never two rounds at once (e.g. a manual refresh during a poll)
      busy = true;
      try {
        const health = await getHealth(ctrl.signal);
        const camera = await getCameraStatus(ctrl.signal);
        setStatus({ health, camera, error: null, loading: false, updatedAt: new Date() });
      } catch (e) {
        if (isAbort(e)) return;
        const error = e instanceof ApiError ? e : new ApiError("invalid", String(e));
        // forget the old values, so nothing stale is shown as current
        setStatus((s) => ({ ...s, health: null, camera: null, error, loading: false }));
      } finally {
        busy = false;
      }
    };
    const loop = async () => {
      await round();
      if (!stopped) timer = window.setTimeout(loop, POLL_MS);
    };

    refreshRef.current = () => void round();
    void loop();
    return () => {
      stopped = true;
      window.clearTimeout(timer);
      ctrl.abort();
    };
  }, []);

  return { ...status, refresh: () => refreshRef.current() };
}
