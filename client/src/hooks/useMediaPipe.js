import { useEffect } from 'react'
import { FilesetResolver, HandLandmarker, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.js"

const useMediaPipe = (webcamRef, canvasRef) => {
    useEffect(() => {
        const initMediaPipe = async () => {
            const vision = await FilesetResolver.forVisionTasks(
                "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
            )
            const handLandmarker = await HandLandmarker.createFromOptions(vision, {
                baseOptions: {
                    modelAssetPath: "/models/hand_landmarker.task"
                },
                numHands: 2,
                runningMode: "video"
            })

            const canvasCtx = canvasRef.current.getContext('2d')
            const drawingUtils = new DrawingUtils(canvasCtx)

            let lastVideoTime = -1

            const renderLoop = () => {
                const video = webcamRef.current?.video

                if (video && video.readyState === 4 && video.currentTime !== lastVideoTime) {
                    canvasRef.current.width = video.videoWidth
                    canvasRef.current.height = video.videoHeight

                    const detections = handLandmarker.detectForVideo(video, Date.now())
                    lastVideoTime = video.currentTime

                    canvasCtx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)

                    if (detections.landmarks) {
                        for (const landmarks of detections.landmarks) {
                            drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, { color: "#00FF00", lineWidth: 5 })
                            drawingUtils.drawLandmarks(landmarks, { color: "#FF0000", lineWidth: 2 })
                        }
                    }
                }

                requestAnimationFrame(renderLoop)
            }

            renderLoop()
        }

        initMediaPipe()
    }, [])
}

export default useMediaPipe