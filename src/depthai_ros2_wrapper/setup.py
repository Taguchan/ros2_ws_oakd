from setuptools import find_packages, setup

package_name = 'depthai_ros2_wrapper'

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
    maintainer='serpens22',
    maintainer_email='serpens22@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rgb_preview = depthai_ros2_wrapper.rgb_preview:main', 
            'stereo_preview = depthai_ros2_wrapper.stereo_preview:main',
            'fer_preview = depthai_ros2_wrapper.fer_preview:main'
        ],
    },
)
