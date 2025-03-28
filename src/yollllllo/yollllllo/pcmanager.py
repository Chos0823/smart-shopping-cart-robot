import sys
import rclpy
from rclpy.node import Node
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QMessageBox
from PyQt5.QtCore import QTimer,QMetaObject,Qt
from PyQt5.QtGui import QFont, QColor, QPalette
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist, Point, Quaternion
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from nav2_msgs.srv import SetInitialPose
from threading import Thread

class RobotUI(Node, QWidget):
    def __init__(self):
        Node.__init__(self, 'robot_ui')
        QWidget.__init__(self)
        self.is_returning = False
        self.is_moving= False
        self.waiting_for_pose_set = False
        # 초기 위치와 복귀 위치 설정
        self.init_pose = [0.0, 0.0, 0.0, 1.0]  # 초기 위치 (x, y, z, w)
        self.return_pose = [0.0, 0.0, 0.0, 1.0]  # 복귀 위치
        
        self.setWindowTitle('관리자용 UI')
        self.setGeometry(100, 100, 500, 400)
        self.setStyleSheet("background-color: #2E2E2E; color: #FFFFFF;")
        
        self.layout = QVBoxLayout()
        
        # 상태 라벨
        self.status_label = QLabel('로봇 시스템 대기중..', self)
        self.status_label.setFont(QFont('Arial', 14, QFont.Bold))
        self.status_label.setStyleSheet("padding: 10px; background-color: #444; border-radius: 10px;")
        self.layout.addWidget(self.status_label)
        
        # 현재 좌표 라벨
        self.position_label = QLabel('현재 좌표: (0.0, 0.0, 0.0, 1.0)', self)
        self.position_label.setFont(QFont('Arial', 12))
        self.position_label.setStyleSheet("padding: 8px; background-color: #555; border-radius: 10px;")
        self.layout.addWidget(self.position_label)

        # 속도 라벨
        self.velocity_label = QLabel('속도: (0.0, 0.0)', self)
        self.velocity_label.setFont(QFont('Arial', 12))
        self.velocity_label.setStyleSheet("padding: 8px; background-color: #555; border-radius: 10px;")
        self.layout.addWidget(self.velocity_label)

        # 별도의 메시지 라벨
        self.return_status_label = QLabel('', self)
        self.return_status_label.setFont(QFont('Arial', 12))
        self.return_status_label.setStyleSheet("padding: 8px; background-color: #666; border-radius: 10px;")
        self.return_status_label.hide() 
        self.layout.addWidget(self.return_status_label)
        
        # 복귀 버튼
        self.return_button = QPushButton('복귀', self)
        self.return_button.setFont(QFont('Arial', 14, QFont.Bold))
        self.return_button.setStyleSheet("background-color: #FF5733; color: white; border-radius: 15px; padding: 10px;")
        self.return_button.setVisible(False)
        self.return_button.clicked.connect(self.return_to_waiting_area)
        self.layout.addWidget(self.return_button)

        self.setLayout(self.layout)
        
        # 구독자
        self.vel_sub = self.create_subscription(Twist, '/cmd_vel', self.vel_callback, 10)
        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.pose_callback, 10)        
        # 네비게이션 액션 클라이언트
        self.nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')  
        # 초기 위치 설정을 위한 서비스 클라이언트
        self.set_initial_pose_service_client = self.create_client(SetInitialPose, "/set_initial_pose") 
        # 초기 위치 설정 요청
        self.set_initial_pose()
        # 좌표 업데이트 속도 설정 (Hz)
        self.rate = self.create_rate(100) 

        self.stop_timer = QTimer(self)
        self.stop_timer.setInterval(10000)  # 10초
        self.stop_timer.timeout.connect(self.show_error_popup)

    def set_initial_pose(self):
        # 초기 위치 설정을 위한 서비스 요청 생성
        req = SetInitialPose.Request()
        req.pose.header.frame_id = 'map'
        
        # PoseWithCovarianceStamped 객체에서 pose를 설정
        req.pose.pose.pose.position = Point(x=0.0, y=0.0, z=0.0)  # x, y 설정
        req.pose.pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=0.1)  # z, w 설정
        req.pose.pose.covariance = [0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0,
                                     0.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01]
        
        # 서비스 요청 보내기
        future = self.set_initial_pose_service_client.call_async(req)
        future.add_done_callback(self.initial_pose_callback)

    def initial_pose_callback(self, future):
        # 초기 위치 설정 결과 처리
        if future.result():
            self.get_logger().info('[INFO] 초기 위치 설정 완료')
            self.status_label.setText('초기 위치 설정 완료')

            if self.waiting_for_pose_set:
                self.waiting_for_pose_set = False
                self.navigate_to_pose(self.return_pose)
            
        else:
            self.get_logger().warn('[WARN] 초기 위치 설정 실패')
            self.status_label.setText('초기 위치 설정 실패')
    
    def vel_callback(self, msg):

        # 속도 값을 표시
        linear_speed = msg.linear.x
        angular_speed = msg.angular.z
        self.velocity_label.setText(f'속도: ({linear_speed:.2f}, {angular_speed:.2f})')
        if self.is_returning:
            return 
        if self.is_returning == False:
            
            if msg.linear.x != 0.0 or msg.angular.z != 0.0:
                self.status_label.setText('상태 : 이동 및 추적')
                self.is_moving = True             
                QMetaObject.invokeMethod(self.stop_timer, "stop", Qt.QueuedConnection)  # 메인 스레드에서 실행
                QMetaObject.invokeMethod(self.stop_timer, "start", Qt.QueuedConnection)  # 메인 스레드에서 실행
            else:
                self.status_label.setText('상태 : 정지')
                if not self.is_moving:
                    QMetaObject.invokeMethod(self.stop_timer, "start", Qt.QueuedConnection)  # 메인 스레드에서 실행
                self.is_moving = False

    def show_error_popup(self):
        """10초 이상 정지 상태일 때 오류 팝업 표시"""
        if not self.is_returning: 
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("로봇 작동 오류!")
            msg_box.setText("오류")
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.buttonClicked.connect(self.reset_timer)  # 확인 후 타이머 재시작
            msg_box.exec_()   

    def reset_timer(self):
        """팝업 확인 후 타이머 리셋"""
        QMetaObject.invokeMethod(self.stop_timer, "stop", Qt.QueuedConnection)  # 메인 스레드에서 실행
        QMetaObject.invokeMethod(self.stop_timer, "start", Qt.QueuedConnection)  # 메인 스레드에서 실행

    def pose_callback(self, msg):
        # 좌표 업데이트
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        z, w = msg.pose.pose.orientation.z, msg.pose.pose.orientation.w
        self.position_label.setText(f'현재 좌표: ({x:.2f}, {y:.2f}, {z:.2f}, {w:.2f})')  # 좌표 및 회전 값 업데이트

        # 현재 위치 저장 (복귀할 때 사용)
        self.current_pose = [x, y, z, w]
        self.pose_received = True  # 위치 정보 수신 완료 표시

        if self.in_target_area(x, y) and self.is_returning == False:
            self.return_status_label.setText('로봇이 계산대에 도착했습니다! 대기장소로 복귀시키겠습니까?')
            self.return_status_label.show()  # 목표 도착 시 표시
            self.return_button.setVisible(True)
        else:
            self.return_status_label.hide()  # 범위를 벗어나면 숨김
            self.return_button.setVisible(False)
        # 빠른 좌표 갱신을 위한 루프
        self.rate.sleep()

    def in_target_area(self, x, y):
        return -0.22 < x < 0.81 and 1.56 < y < 2.25  # 계산대 범위 지정
    
    def return_to_waiting_area(self):
        self.is_returning = True  # 복귀 중 플래그 설정
        self.status_label.setText('로봇이 대기장소로 복귀중...')
        self.return_button.setVisible(False)
        self.return_status_label.hide()
        self.waiting_for_pose_set = True
        if self.pose_received:
            self.navigate_to_pose([0.0, 0.0, 0.0, 1.0])

    
    
    def navigate_to_pose(self, pose):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        
        # 목표 위치 설정
        goal_msg.pose.pose.position.x = pose[0]  # x
        goal_msg.pose.pose.position.y = pose[1]  # y
        goal_msg.pose.pose.position.z = 0.0      # z, 필요한 경우 0으로 설정
        
        # 목표 회전 설정
        goal_msg.pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=pose[2], w=pose[3])
        
        # 서버가 준비될 때까지 기다립니다.
        self.nav_client.wait_for_server()

        # 목표 위치로 이동 요청
        send_goal_future = self.nav_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

     
    
    def goal_response_callback(self, future):
        goal_handle = future.result()
        if goal_handle.accepted:
            self.get_logger().info('이동 명령 수락됨')
        else:
            self.get_logger().info('이동 명령 거부됨')
        # 이동 결과를 받아봅니다.
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result_callback)
    
    def goal_result_callback(self, future):
        self.is_returning = False  # 복귀 중 플래그 해제
        result = future.result()

        if result.result == 'SUCCEEDED':
            # 목표 도달시 팝업 띄우기
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("알림")
            msg_box.setText("로봇이 대기장소에 도착했습니다!")
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.buttonClicked.connect(self.close_popup)
            msg_box.exec_()
        else:
            self.get_logger().warn('목표 위치 도달 실패')
    
    def close_popup(self, i):
        pass  # 팝업 확인 후 아무 동작도 안함

def main():
    rclpy.init()
    app = QApplication(sys.argv)
    robot_ui = RobotUI()
    robot_thread = Thread(target=rclpy.spin, args=(robot_ui,), daemon=True)
    robot_thread.start()
    robot_ui.show()
    app.exec_()

if __name__ == '__main__':
    main()
