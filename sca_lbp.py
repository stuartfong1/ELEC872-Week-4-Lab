import mediapipe as mp
import numpy as np
from scipy import ndimage

face_detector = mp.solutions.face_detection.FaceDetection()

def sca_lbp(image: np.ndarray) -> list[tuple[int]]:
    """Stuart's Crazy Awesome LBP (SCA-LBP)."""
    rgb_image = np.stack((image, image, image), axis=-1).astype(np.uint8)

    height, width = image.shape

    result = face_detector.process(rgb_image)

    coordinates = []

    if result.detections:
        landmarks = result.detections[0].location_data.relative_keypoints

        for landmark in landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)

            coordinates.append((y, x))

    else:
        msg = "Womp womp. Detection failed."
        raise ValueError(msg)

    # Use the intensities of the landmarks to correct the lighting
    # Get the median which is hopefully the person's skin
    left_intensity = np.median([
        image[coordinates[-2][0] - 10 : coordinates[-2][0] + 20, coordinates[-2][1] + 2 : coordinates[-2][1] + 20],
    ]) / 255
    right_intensity = np.median([
        image[coordinates[-1][0] - 10 : coordinates[-1][0] + 20, coordinates[-1][1] - 20 : coordinates[-1][1] - 1],
    ]) / 255

    # Don't bother correcting if the difference is small
    # The estimate is quite noisy
    if np.abs(left_intensity - right_intensity) > 30 / 255:
        image = np.clip(image * np.linspace(right_intensity, left_intensity, width), 0, 255).astype(np.uint8)

    image_center = (round(height / 2), round(width / 2))
    left_coord = (coordinates[-2][0] - image_center[0], coordinates[-2][1] - image_center[1])
    right_coord = (coordinates[-1][0] - image_center[0], coordinates[-1][1] - image_center[1])

    # Rotate the image so that the y-coordinates are aligned
    slope = (left_coord[0] - right_coord[0]) / (left_coord[1] - right_coord[1])
    angle = np.arctan(slope)
    image = ndimage.rotate(image, angle * 180 / np.pi, reshape=False, order=1, mode="nearest")
    # rotation_matrix = np.array([
    #                            [np.cos(angle), np.sin(angle)],
    #                            [-np.sin(angle), np.cos(angle)],
    #                        ])
    # rotated_coordinates = np.round(np.array(coordinates) @ rotation_matrix).astype(np.uint8)

    # # Flip the image, then re-center and take the mean
    # face_center = np.round((rotated_coordinates[-2, 1] + rotated_coordinates[-1, 1]) / 2)
    # reflection_matrix = np.array([[1, 0], [0, -1]])
    # bias = np.array((0, 2 * face_center))
    # image = (image + ndimage.affine_transform(
    #     image,
    #     reflection_matrix,
    #     bias,
    # )) / 2


    return image
