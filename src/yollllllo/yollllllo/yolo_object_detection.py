import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import os
import threading

class YOLOTrackingPublisher(Node):
    def __init__(self):
        super().__init__('yolo_tracking_publisher')
        # 'AMR_image' 토픽으로 CompressedImage 메시지를 퍼블리시할 퍼블리셔 생성
        self.publisher_ = self.create_publisher(CompressedImage, 'AMR_image', 10)
        # 'cmd_vel' 토픽으로 Twist 메시지를 퍼블리시할 퍼블리셔 생성('AMR_image'토픽과 동시 처리 위해 스레드 필수)
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10) 
        self.bridge = CvBridge()
        # YOLO 모델 로드
        self.model = YOLO('/home/johyunsuk/wlwjdq/src/yollllllo/best.pt')
        # 웹캠 영상 캡처를 위한 VideoCapture 객체 생성
        self.camera_index = 0
        self.max_camera_index = 5
        self.cap = None
        self.boundery = 0
        self.initialize_camera()
        # 타이머를 설정하여 주기적으로 timer_callback 함수 호출
        self.timer = self.create_timer(0.1, self.timer_callback)  # 주기 조정 가능
        self.carheight = None # 객체인식된 자동차 높
        self.dum_height = None # 객체인식된 더미 높이
        self.obstacle_avoiding = False  # 장애물 회피 상태 플래그
        self.obstacle_timer = None     # 회피 동작 종료 타이머
    
    def initialize_camera(self):
        while self.camera_index < self.max_camera_index:
            self.cap = cv2.VideoCapture(self.camera_index)
            if self.cap.isOpened():
                self.get_logger().info(f"카메라 {self.camera_index} 초기화 성공")
                return
            else:
                self.get_logger().warn(f"카메라 {self.camera_index} 초기화 실패, 다음 카메라 시도")
                self.camera_index += 1

        self.get_logger().error("사용 가능한 카메라를 찾을 수 없습니다.")

    def timer_callback(self):
        # 웹캠에서 프레임을 읽어옴
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("웹캠이 인식되지 않아 프레임 캡쳐에 실패했습니다...")  # 프레임 캡처 실패 시 경고
            return

        # 카메라 중심점 x값 계산
        camera_center_x = frame.shape[1] // 2

        # YOLO 모델을 사용하여 객체 추적 및 결과 반환
        results = self.model.track(source=frame, show=False, tracker='bytetrack.yaml')

        # carcenter_x, carcenter_y, dumcenter_x 변수 초기화 (기본값 None)
        carcenter_x = None
        carcenter_y = None
        dumcenter_x = None
        #속도제어 메시지 객체 생성
        twist = Twist()

        # 결과에서 각각의 검출된 객체 데이터를 처리
        for result in results:
            for detection in result.boxes.data:
                #self.get_logger().info(f"객체 데이터: {detection}")  # 검출된 객체 데이터 출력

                # 검출된 데이터가 최소 6개 요소를 가질 경우 (x1, y1, x2, y2, confidence, class_id)



                if len(detection) < 7:
                    detection = list(detection) + [100] * (7 - len(detection))
                if len(detection) >= 6:
                    x1, y1, x2, y2, track_id, confidence, class_id = detection[:7]

                    # 클래스 confidence가 0.85 미만인 객체는 건너뜁니다.
                    class_confidence = detection[5]  # 클래스 confidence 값
                    if class_confidence < 0.85:
                        continue  # confidence가 0.85 미만이면 해당 객체를 처리하지 않음
                    
                    # 클래스 ID가 1인 경우, dum_height 계산
                    if int(class_id) == 1:
                        dumcenter_x = int((x1 + x2) / 2)
                        self.dum_height = int(y2 - y1)  # bounding box의 높이를 dum_height에 저장
                        self.get_logger().info(f"dum_height: {self.dum_height}")
                        # 장애물 회피 중이라면 회피 작업 수행                       
                        if self.obstacle_avoiding:
                            twist = Twist()
                            self.avoid_obstacle(twist, dumcenter_x, camera_center_x,frame,x1, y1, x2, y2, track_id, confidence, class_id)
                            return  
                        # 장애물을 인식했는데 충돌할 위험이 있음에도 불구하고 장애물 회피 중이 아닐 때
                        elif self.obstacle_avoiding==False and self.dum_height>=300:  
                            self.get_logger().info("장애물이 탐지되었습니다!! 회피를 시작합니다...")
                            self.obstacle_avoiding = True
                            self.avoid_obstacle(twist, dumcenter_x, camera_center_x,frame,x1, y1, x2, y2, track_id, confidence, class_id)
                            self.start_obstacle_timer()  # 회피 상태 타이머 시작
                            break


                    # 클래스 ID와 테이블 ID 확인
                    if int(class_id) == 0 and int(track_id) == 1:
                        # 객체의 중심 좌표 계산
                        carcenter_x = int((x1 + x2) / 2)
                        carcenter_y = int((y1 + y2) / 2)
                        # self.get_logger().info(f"Detection 데이터: {detection}")
                        self.boundery = detection[3]
                        # self.get_logger().info(f"Updated self.boundery: {self.boundery}")                 

                        # angular_edit 계산
                        angular_edit = carcenter_x - camera_center_x
                        self.get_logger().info(f"angular_edit: {angular_edit}")

                        self.carheight = int(y2 - y1)
                        self.get_logger().info(f'carheight: {self.carheight}')

                        # 장애물 회피가 완료된 후에만 track_id=1인 자동차의 adjust_speed와 adjust_angular 실행
                        if self.carheight is not None and not self.obstacle_avoiding:
                            self.adjust_speed(self.carheight,twist)
                            self.adjust_angular(angular_edit,twist)
                        

                    # 프레임에 바운딩 박스와 중심점 그리기
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                    # 중심점이 정의된 경우에만 원을 그림
                    if carcenter_x is not None and carcenter_y is not None:
                        cv2.circle(frame, (carcenter_x, carcenter_y), 5, (0, 0, 255), -1)

                    label_text = f'Conf: {confidence:.2f} Class: {int(class_id)}'
                    if track_id is not None:
                        label_text = f'Track_ID: {int(track_id)}, ' + label_text

                    # 프레임에 텍스트로 라벨 추가
                    cv2.putText(frame, label_text, (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 프레임을 압축하여 ROS 2에서 사용할 수 있는 CompressedImage 메시지로 변환
        _, compressed_frame = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        compressed_img_msg = CompressedImage()
        compressed_img_msg.header.stamp = self.get_clock().now().to_msg()  # 현재 시간 스탬프
        compressed_img_msg.format = "jpeg"
        compressed_img_msg.data = compressed_frame.tobytes()

        # 압축된 이미지를 퍼블리시
        self.publisher_.publish(compressed_img_msg)
        # cmd_vel 메시지 퍼블리시
        self.cmd_vel_publisher.publish(twist)


    
    
    
    
    
    def avoid_obstacle(self, twist, dumcenter_x, camera_center_x, frame, x1, y1, x2, y2, track_id, confidence, class_id):
        """
        장애물 회피 로직: 장애물의 위치에 따라 우회 방향 설정
        """
        if dumcenter_x < camera_center_x:  # 장애물이 왼쪽에 있으면 오른쪽으로 이동
            twist.linear.x = 0.1  # 천천히 직진
            twist.angular.z = -0.2  # 오른쪽으로 회전
            self.get_logger().info("장애물 회피: 우회전중...")
           
        else:  # 장애물이 오른쪽에 있으면 왼쪽으로 이동
            twist.linear.x = 0.1
            twist.angular.z = 0.2
            self.get_logger().info("장애물 회피: 좌회전중...")
         
         # 장애물 회피 동안 cmd_vel 메시지를 지속적으로 퍼블리시
        self.cmd_vel_publisher.publish(twist)

        # 프레임에 바운딩 박스와 중심점 그리기
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)


        label_text = f'Conf: {confidence:.2f} Class: {int(class_id)}'
        if track_id is not None:
            label_text = f'Track_ID: {int(track_id)}, ' + label_text

        # 프레임에 텍스트로 라벨 추가
        cv2.putText(frame, label_text, (int(x1), int(y1) - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 회피 중인 이미지를 압축하여 퍼블리시
        _, compressed_frame = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        compressed_img_msg = CompressedImage()
        compressed_img_msg.header.stamp = self.get_clock().now().to_msg()  # 현재 시간 스탬프
        compressed_img_msg.format = "jpeg"
        compressed_img_msg.data = compressed_frame.tobytes()

        # 압축된 이미지를 퍼블리시
        self.publisher_.publish(compressed_img_msg)
          
           
    

    def start_obstacle_timer(self):
        """
        회피 동작 후 장애물 무시 시간을 설정
        """
        # 이미 타이머가 실행 중인 경우, 새로 시작하지 않음
        if  self.obstacle_timer is not None and self.obstacle_timer.is_alive():
                return
            # 3초 동안 장애물 무시(별도의 스레드에서 진행)
        self.obstacle_timer = threading.Timer(3.0, self.reset_obstacle_avoidance) 
        self.obstacle_timer.start()

    def reset_obstacle_avoidance(self):
        """
        회피 상태 플래그를 초기화하여 장애물 감지를 재개
        """
        self.obstacle_avoiding = False
        self.get_logger().info("장애물 회피 완료, 객체 정상 추적중...")
    
    def adjust_speed(self, height, twist):
        """
        객체의 높이에 따라 전진 속도를 세밀하게 조정.
        """
        if 470 < self.boundery < 485:
            speed = 0.0  # 정지
            # print(f"self.bound:{self.boundery}")
            self.get_logger().info("box가 닿을정도로 가까움 : 멈춤")
        elif height > 280:  # 매우 가까움
            speed = 0.0  # 정지
            self.get_logger().info("거리가 너무 가까움: 멈춤")
        elif 200 < height <= 280:  # 가까움
            speed = 0.05  # 매우 느린 속도
            self.get_logger().info("거리가 가까움: 매우천천히")
        elif 170 < height <= 200:  # 중간 거리
            speed = 0.1  # 느린 속도
            self.get_logger().info("거리가 조금 가까움: 천천히")
        elif 125 < height <= 170:  # 적정 거리
            speed = 0.15  # 적당한 속도
            self.get_logger().info("거리가 적당함: 표준속도")
        elif 100 < height <= 125:  # 먼 거리
            speed = 0.2  # 빠른 속도
            self.get_logger().info("거리가 멈: 빠르게")
        else:  # 매우 먼 거리
            speed = 0.3  # 최대 속도
            self.get_logger().info("거리가 매우 멈: 매우 빠르게")

        twist.linear.x = speed  # Twist 메시지에 선형 속도 설정


    def adjust_angular(self, angular_edit, twist):
        """
        중심점 차이에 따라 좌/우회전을 세밀하게 조정.
        """
        if angular_edit > 200:  # 중심점이 오른쪽으로 크게 치우침
            twist.angular.z = -0.2  # 빠른 우회전
            self.get_logger().info("빠른 우회전")
        elif 150 < angular_edit <= 200:  # 중심점이 오른쪽으로 약간 치우침
            twist.angular.z = -0.1  # 중간 우회전
            self.get_logger().info("중간 우회전")
        elif 20 < angular_edit <= 150:  # 중심점이 오른쪽으로 조금 치우침
            twist.angular.z = -0.05  # 느린 우회전
            self.get_logger().info("느린 우회전")
        elif -20 <= angular_edit <= 20:  # 중심점이 거의 정중앙
            twist.angular.z = 0.0  # 회전 없음
            self.get_logger().info("회전 없음")
        elif -150 <= angular_edit < -20:  # 중심점이 왼쪽으로 조금 치우침
            twist.angular.z = 0.05  # 느린 좌회전
            self.get_logger().info("느린 좌회전")
        elif -200 <= angular_edit < -150:  # 중심점이 왼쪽으로 약간 치우침
            twist.angular.z = 0.1  # 중간 좌회전
            self.get_logger().info("중간 좌회전")
        else:  # 중심점이 왼쪽으로 크게 치우침
            twist.angular.z = 0.2  # 빠른 좌회전
            self.get_logger().info("빠른 좌회전")
    

        
    def destroy_node(self):
        # 노드 종료 시 웹캠 리소스 해제
        super().destroy_node()
        self.cap.release()   



def main(args=None):
    # ROS 2 초기화
    rclpy.init(args=args)
    # YOLOTrackingPublisher 노드 생성
    node = YOLOTrackingPublisher()
    
    # 멀티스레딩을 사용하여 이미지 퍼블리셔와 cmd_vel 퍼블리셔 동시에 실행
    thread = threading.Thread(target=rclpy.spin, args=(node,))
    thread.start()

    # 노드 종료 대기
    thread.join()

    # 노드 종료
    node.destroy_node()
    # OpenCV 윈도우 종료
    cv2.destroyAllWindows()
    # ROS 2 종료
    rclpy.shutdown()

if __name__ == '__main__':
    main()

# 이미지가 짤리는 경우에 로직을 구현 로봇을 그때 어떻게 구현할 것인지
# 테스팅? 통신이 안되면 ros2과의 통신이 안됨 일정기간  토픽이 안들어오면 다른 토픽이나 오류 탐지 방법
# 장애물이 크면 following을 끄고 navi on
# 장애물이 움직이는 경우 작은박스가 있으면 돌릴수도 있지만 height를 가지고 거리도 생각할수 있다.






