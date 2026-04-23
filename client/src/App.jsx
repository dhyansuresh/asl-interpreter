import {useRef} from "react";
import Webcam from "react-webcam";
import useMediaPipe from './hooks/useMediaPipe'



function App() {
    const webcamRef = useRef(null);
    const canvasRef = useRef(null)
    useMediaPipe(webcamRef, canvasRef);

  return (
      <div>
          <canvas ref={canvasRef}></canvas>
          <Webcam ref={webcamRef} />
      </div>
  )
}

export default App
