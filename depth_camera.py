import depthai as dai
import cv2

with dai.Pipeline() as pipeline:
    # LEFT mono
    left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)

    # RIGHT mono
    right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    # RGB center
    rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)

    # Output queues
    q_left = left.requestOutput(
        size=(640, 400), type=dai.ImgFrame.Type.GRAY8, fps=30
    ).createOutputQueue()

    q_right = right.requestOutput(
        size=(640, 400), type=dai.ImgFrame.Type.GRAY8, fps=30
    ).createOutputQueue()

    q_rgb = rgb.requestOutput(
        size=(1280, 720), type=dai.ImgFrame.Type.BGR888i, fps=30
    ).createOutputQueue()

    # Start pipeline
    pipeline.start()

    # Video writers
    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    writer_left = cv2.VideoWriter("left.avi", fourcc, 30, (640, 400), False)
    writer_right = cv2.VideoWriter("right.avi", fourcc, 30, (640, 400), False)
    writer_rgb = cv2.VideoWriter("rgb.avi", fourcc, 30, (1280, 720), True)

    print("Nagrywanie... q = stop")

    while pipeline.isRunning():
        frame_left = q_left.get().getCvFrame()
        frame_right = q_right.get().getCvFrame()
        frame_rgb = q_rgb.get().getCvFrame()

        # Save
        writer_left.write(frame_left)
        writer_right.write(frame_right)
        writer_rgb.write(frame_rgb)

        # Preview
        cv2.imshow("LEFT", frame_left)
        cv2.imshow("RIGHT", frame_right)
        cv2.imshow("RGB", frame_rgb)

        if cv2.waitKey(1) == ord("q"):
            break

    writer_left.release()
    writer_right.release()
    writer_rgb.release()

cv2.destroyAllWindows()
