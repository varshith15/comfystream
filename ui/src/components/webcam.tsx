import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Internal component that renders and captures camera feed at exactly 512x512.
 * Handles both display and stream capture in a single canvas element,
 * ensuring consistent dimensions while maintaining aspect ratio.
 */
function StreamCanvas({
  stream,
  frameRate,
  onStreamReady,
}: {
  stream: MediaStream | null;
  frameRate: number;
  onStreamReady: (stream: MediaStream) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current!;
    const outputStream = canvas.captureStream(30);
    onStreamReady(outputStream);

    return () => {
      outputStream.getTracks().forEach((track) => track.stop());
    };
  }, [onStreamReady, frameRate]);

  // Set up canvas animation loop
  useEffect(() => {
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext("2d")!;

    let isActive = true;
    const drawFrame = () => {
      if (!isActive) {
        // return without scheduling another frame
        return;
      }
      const video = videoRef.current!;
      if (!video?.videoWidth) {
        // video is not ready yet
        requestAnimationFrame(drawFrame);
        return;
      }

      const scale = Math.max(512 / video.videoWidth, 512 / video.videoHeight);

      const scaledWidth = video.videoWidth * scale;
      const scaledHeight = video.videoHeight * scale;

      const offsetX = (512 - scaledWidth) / 2;
      const offsetY = (512 - scaledHeight) / 2;

      ctx.fillStyle = "black";
      ctx.fillRect(0, 0, 512, 512);
      ctx.drawImage(video, offsetX, offsetY, scaledWidth, scaledHeight);
      requestAnimationFrame(drawFrame);
    };
    drawFrame();

    return () => {
      isActive = false;
    };
  }, []);

  useEffect(() => {
    if (!stream) return;
    if (!videoRef.current) {
      videoRef.current = document.createElement("video");
      videoRef.current.muted = true;
    }

    const video = videoRef.current;
    video.srcObject = stream;
    video.onloadedmetadata = () => {
      video.play().catch((error) => {
        console.log("Video play failed:", error);
      });
    };

    return () => {
      video.pause();
      video.srcObject = null;
    };
  }, [stream]);

  return (
    <canvas
      ref={canvasRef}
      width={512}
      height={512}
      className="w-full h-full"
      style={{
        backgroundColor: "black",
      }}
    />
  );
}

interface WebcamProps {
  onStreamReady: (stream: MediaStream) => void;
  deviceId: string;
  frameRate: number;
  selectedAudioDeviceId: string;
}

export function Webcam({ onStreamReady, deviceId, frameRate, selectedAudioDeviceId }: WebcamProps) {
  const [stream, setStream] = useState<MediaStream | null>(null);

  const replaceStream = useCallback((newStream: MediaStream | null) => {
    setStream((oldStream) => {
      // Clean up old stream if it exists
      if (oldStream) {
        oldStream.getTracks().forEach((track) => track.stop());
      }
      if (newStream) {
        const videoTrack = newStream.getVideoTracks()[0];
        const settings = videoTrack.getSettings();
      }
      return newStream;
    });
  }, []);

  const startWebcam = useCallback(async () => {
    if (!deviceId || !selectedAudioDeviceId) {
      return null;
    }
    if (frameRate == 0) {
      return null;
    }

    try {
      const newStream = await navigator.mediaDevices.getUserMedia({
        video: {
          ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
          width: { ideal: 512 },
          height: { ideal: 512 },
          aspectRatio: { ideal: 1 },
          frameRate: { ideal: frameRate, max: frameRate },
        },
        audio: {
          ...(selectedAudioDeviceId ? { deviceId: { exact: selectedAudioDeviceId } } : {}),
          sampleRate: 16000,
          sampleSize: 16,
          channelCount: 1,
        },
      });
      return newStream;
    } catch (error) {
      console.error("Error accessing media devices.", error);
      return null;
    }
  }, [deviceId, frameRate, selectedAudioDeviceId]);

  useEffect(() => {
    if (!deviceId || !selectedAudioDeviceId) return;
    if (frameRate == 0) return;

    startWebcam().then((newStream) => {
      replaceStream(newStream);
      if (newStream) {
        onStreamReady(newStream);
      }
    });

    return () => {
      replaceStream(null);
    };
  }, [deviceId, frameRate, selectedAudioDeviceId, startWebcam, replaceStream, onStreamReady]);

  return (
    <div>
      <StreamCanvas
        stream={stream}
        frameRate={frameRate}
        onStreamReady={onStreamReady}
      />
    </div>
  );
}
