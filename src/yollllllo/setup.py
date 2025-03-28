from setuptools import find_packages, setup

package_name = 'yollllllo'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jjb',
    maintainer_email='jjb@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'image_subscriber = yollllllo.image_subscriber:main',  # 실행 명령
            'yolo_object_detection = yollllllo.yolo_object_detection:main',
            'pc_node = yollllllo.pcmanager:main',  # 노드 실행 명령
        ],
    },
)
    
