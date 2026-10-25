import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'wall_follower'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),

    data_files=[
        ('share/ament_index/resource_index/packages', 
            ['resource/wall_follower']),
        ('share/wall_follower', ['package.xml']),
        (os.path.join('share', 'wall_follower', 'launch'), glob('launch/*.py')),
        (os.path.join('share', 'wall_follower', 'config'), glob('config/*.yaml')),
        (os.path.join('share', 'wall_follower', 'worlds'), glob('worlds/*.world')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='ragibyeahsar@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        	'wall_follower_node = wall_follower.wall_follower_node:main',
        	'wall_follower_single_pid = wall_follower.wall_follower_single_pid:main',
                'data_logger = wall_follower.data_logger:main',
        ],
    },
)
