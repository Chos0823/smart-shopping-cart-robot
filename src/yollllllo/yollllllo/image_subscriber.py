import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        # 'AMR_image' 토픽에서 압축된 이미지를 구독합니다.
        self.create_subscription(
            CompressedImage,
            'AMR_image',  # 이미지가 퍼블리시되는 토픽 이름
            self.image_callback,
            10  # 큐 사이즈
        )

        self.bridge = CvBridge()

    def image_callback(self, msg):
        # ROS 메시지를 OpenCV 이미지로 변환
        try:
            # CompressedImage는 JPEG 포맷이므로 이를 디코딩하여 OpenCV 이미지로 변환합니다.
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # OpenCV로 이미지를 화면에 표시
            cv2.imshow("Received Image", cv_image)
            cv2.waitKey(1)  # 1ms 대기, 이 코드가 없으면 화면이 깜빡일 수 있습니다.
        except Exception as e:
            self.get_logger().error(f"이미지 처리 중 오류 발생: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriber()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    # 자원 해제
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
