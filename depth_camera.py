
import depthai as dai
import cv2

with dai.Pipeline() as pipeline:
    cam_left = pipeline.create(dai.node.Camera)
    cam_right = pipeline.create(dai.node.Camera)

    cam_left.build(dai.CameraBoardSocket.CAM_B)  # lewa
    cam_right.build(dai.CameraBoardSocket.CAM_C)  # prawa

    out_left = cam_left.requestOutput((640, 400), dai.ImgFrame.Type.GRAY8)
    out_right = cam_right.requestOutput((640, 400), dai.ImgFrame.Type.GRAY8)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer_left = cv2.VideoWriter("left.avi", fourcc, 30.0, (640, 400))
    writer_right = cv2.VideoWriter("right.avi", fourcc, 30.0, (640, 400))

    # W v3 Device tworzy się bez pipeline, potem start()
    with dai.Device() as device:
        pipeline.start(device)
        print("Nagrywanie... Naciśnij 'q' aby zatrzymać.")

        while pipeline.isRunning():
            frame_left = out_left.get().getCvFrame()
            frame_right = out_right.get().getCvFrame()

            frame_left_bgr = cv2.cvtColor(frame_left, cv2.COLOR_GRAY2BGR)
            frame_right_bgr = cv2.cvtColor(frame_right, cv2.COLOR_GRAY2BGR)

            writer_left.write(frame_left_bgr)
            writer_right.write(frame_right_bgr)

            cv2.imshow("Lewy", frame_left)
            cv2.imshow("Prawy", frame_right)

            if cv2.waitKey(1) == ord("q"):
                break

writer_left.release()
writer_right.release()
cv2.destroyAllWindows()
