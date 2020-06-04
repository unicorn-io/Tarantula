import os 
import time
import pybullet as p
import pybullet_data
import numpy as np
import math
import gym 
from gym.utils import seeding
from gym import error, spaces, utils


class HexSimulator(gym.Env):

    metadata = {'render.modes': ['human']}
    def __init__(self, pathToUrdf=".\\pexod.urdf"):
        p.connect(p.GUI)
        self.GRAVITY = -9.81
        self.dt = 1/240
        self.angles = np.zeros(18)
        
        p.resetDebugVisualizerCamera(cameraDistance=1,
                                                cameraYaw = 20,
                                                cameraPitch=-20,
                                                cameraTargetPosition=[1, -0.5, 0.8])
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()


    def reset(self, pathToUrdf=".\\pexod.urdf"):
        p.resetSimulation()
        p.setGravity(0,0,self.GRAVITY)
        p.setTimeStep(1./240.)
        p.setPhysicsEngineParameter(fixedTimeStep=self.dt)
        self.planeid = p.loadURDF('plane.urdf')

        start_pos = [0,0,0.3]
        start_orientation = p.getQuaternionFromEuler([0.,0,0])
        #self.bot = p.loadURDF("./urdf/pexod.urdf", start_pos, start_orientation)
        self.bot = p.loadURDF(pathToUrdf, start_pos, start_orientation)
        self.joint_list =  self._make_joint_list(self.bot)
        print(self.joint_list)

        p.setRealTimeSimulation(0)
        jointFrictionForce=1
        for joint in range(p.getNumJoints(self.bot)):
            p.setJointMotorControl2(self.bot, joint, p.POSITION_CONTROL, force=jointFrictionForce)
        for t in range(0, 100):
            p.stepSimulation()
            p.setGravity(0,0,self.GRAVITY)
        
        print("Link Ids:")
        self.link_ids = self.get_end_effector_ids()
        print(p.getLinkState(self.bot, self.link_ids[1]))
        self.link_ids = sorted(self.link_ids)
        self.init_position = self.get_pos()[0]
        self.init_orientation = self.get_pos()[1]

    def get_pos(self):
        '''
            Returns the position list of 3 floats and orientation as list of 4 floats in [x,y,z,w] order.
            Use p.getEulerFromQuaternion to convert the quaternion to Euler if needed.
        '''
        return p.getBasePositionAndOrientation(self.bot)

    def step(self, action):
        ''' One step of simulation
            action: 3Vec list of the end effector posiitons.'''
        final_positions = []
        joint_poses = np.zeros(18)
        dv = 0.005
        for k in range(len(self.link_ids)):
            temp_pos = p.getLinkState(self.bot, self.link_ids[k])
            final_positions.append((temp_pos[0][0] + action[k][0]*dv,
                                    temp_pos[0][1] + action[k][1]*dv,
                                    temp_pos[0][2] + action[k][2]*dv))
            q = p.calculateInverseKinematics(self.bot, self.link_ids[k], final_positions[k])
            joint_poses[self.link_ids[k]-2] = q[0]
            joint_poses[self.link_ids[k]-1] = q[1]
            joint_poses[self.link_ids[k]] = q[0]
        p.setJointMotorControlArray(self.bot, range(18), p.POSITION_CONTROL, list(joint_poses))
        #joint_poses = p.calculateInverseKinematics2(self.bot, self.link_ids, final_positions)
        p.stepSimulation()
        current_position = self.get_pos()[0]
        dis_x_direction = current_position[0]-self.init_position[0]
        dis_y_direction = current_position[1]-self.init_position[1]
        dis_z_direction = current_position[2]-self.init_position[2]
        orientation = self.get_pos()[1]
        tilt_translation = math.sqrt(dis_y_direction**2+dis_z_direction**2+
                                    orientation[0]**2+orientation[1]**2+
                                    orientation[2]**2)
        reward = dis_x_direction / tilt_translation
        self.init_position = current_position
        state_object = p.getBasePositionAndOrientation(self.bot)
        state_end_effectors = p.getLinkStates(self.bot, self.link_ids)
        done = True
        if (reward <= 0.00001):
            done == False
        observation = []
        for k in self.link_ids:
            observation.append(p.getLinkState(self.bot, k))
        info = state_object
        return observation, reward, done, info

    def render(self, mode='human') :
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=self.get_pos()[0],
                                                        distance = 1,
                                                        yaw = 20,
                                                        pitch = 20,
                                                        roll = 0,
                                                        upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                     aspect=float(960) /720,
                                                     nearVal=0.1,
                                                     farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(width=960,
                                              height=720,
                                              viewMatrix=view_matrix,
                                              projectionMatrix=proj_matrix,
                                              renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720,960, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def close(self):
        p.disconnect
    



    def _make_joint_list(self,  bot):
        joint_names = [b'body_leg_0', b'leg_0_1_2', b'leg_0_2_3',
        b'body_leg_1', b'leg_1_1_2', b'leg_1_2_3',
        b'body_leg_2', b'leg_2_1_2', b'leg_2_2_3',
        b'body_leg_3', b'leg_3_1_2', b'leg_3_2_3',
        b'body_leg_4', b'leg_4_1_2', b'leg_4_2_3',
        b'body_leg_5', b'leg_5_1_2', b'leg_5_2_3',
        ] 
        joint_list = []
        for n in joint_names:
            joint_found = False
            for joint in range(p.getNumJoints(bot)):
                name = p.getJointInfo(bot, joint)[1]
                if name == n:
                    joint_list += [joint]
                    joint_found = True
            if(joint_found==False):
                joint_list += [-1]
        return joint_list

    def get_joint_pos(self):
        ''' The position value of this joint (joint angle in quaternion form)'''
        pos = np.zeros(len(self.joint_list))
        i = 0
        for j in self.joint_list:
            pos[i] = p.getJointState(self.bot, j)[0]
            i += 1
        return pos

    def get_end_effector_ids(self):
        ''' Link inx == Joint inx from the documentation'''
        # we will be considering only the end joints and thier links
        # leg_0_2_3, leg_1_2_3, leg_2_2_3, leg_3_2_3, leg_4_2_3, leg_5_2_3
        joint_names = [b'leg_0_2_3',b'leg_1_2_3',b'leg_2_2_3',
        b'leg_3_2_3', b'leg_4_2_3',b'leg_5_2_3'] 
        joint_list = []
        for n in joint_names:
            joint_found = False
            for joint in range(p.getNumJoints(self.bot)):
                name = p.getJointInfo(self.bot, joint)[1]
                if name == n:
                    joint_list += [joint]
                    joint_found = True
            if(joint_found==False):
                joint_list += [-1]
        return joint_list

def test_pod():
    simu = HexSimulator(".\\pexod.urdf")
    for i in range(10000):
        if (i < 10): simu.step([(0,0,1000),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0)])
        else: simu.step([(1000,0,-1000),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0)])
        simu.render()
        time.sleep(1./240.)
    


if __name__ == "__main__":
    test_pod()
    