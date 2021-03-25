import numpy as np
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.rail_env import fast_count_nonzero, fast_argmax


"""
LICENCE for the FastTreeObs Observation Builder

The observation can be used freely and reused for further submissions. Only the author needs to be referred to
/mentioned in any submissions - if the entire observation or parts, or the main idea is used.

Author: Adrian Egli (adrian.egli@gmail.com)

[Linkedin](https://www.researchgate.net/profile/Adrian_Egli2)
[Researchgate](https://www.linkedin.com/in/adrian-egli-733a9544/)
"""

class FastTreeObs(ObservationBuilder):

    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.observation_dim = 26

    def build_data(self):
        if self.env is not None:
            self.env.dev_obs_dict = {}
        self.switches = {}
        self.switches_neighbours = {}
        self.debug_render_list = []
        self.debug_render_path_list = []
        if self.env is not None:
            self.find_all_cell_where_agent_can_choose()

    def find_all_cell_where_agent_can_choose(self):
        switches = {}
        for h in range(self.env.height):
            for w in range(self.env.width):
                pos = (h, w)
                for dir in range(4):
                    possible_transitions = self.env.rail.get_transitions(*pos, dir)
                    num_transitions = fast_count_nonzero(possible_transitions)
                    if num_transitions > 1:
                        if pos not in switches.keys():
                            switches.update({pos: [dir]})
                        else:
                            switches[pos].append(dir)

        switches_neighbours = {}
        for h in range(self.env.height):
            for w in range(self.env.width):
                # look one step forward
                for dir in range(4):
                    pos = (h, w)
                    possible_transitions = self.env.rail.get_transitions(*pos, dir)
                    for d in range(4):
                        if possible_transitions[d] == 1:
                            new_cell = get_new_position(pos, d)
                            if new_cell in switches.keys() and pos not in switches.keys():
                                if pos not in switches_neighbours.keys():
                                    switches_neighbours.update({pos: [dir]})
                                else:
                                    switches_neighbours[pos].append(dir)

        self.switches = switches
        self.switches_neighbours = switches_neighbours

    def check_agent_decision(self, position, direction):
        switches = self.switches
        switches_neighbours = self.switches_neighbours
        agents_on_switch = False
        agents_on_switch_all = False
        agents_near_to_switch = False
        agents_near_to_switch_all = False
        if position in switches.keys():
            agents_on_switch = direction in switches[position]
            agents_on_switch_all = True

        if position in switches_neighbours.keys():
            new_cell = get_new_position(position, direction)
            if new_cell in switches.keys():
                if not direction in switches[new_cell]:
                    agents_near_to_switch = direction in switches_neighbours[position]
            else:
                agents_near_to_switch = direction in switches_neighbours[position]

            agents_near_to_switch_all = direction in switches_neighbours[position]

        return agents_on_switch, agents_near_to_switch, agents_near_to_switch_all, agents_on_switch_all

    def required_agent_decision(self):
        agents_can_choose = {}
        agents_on_switch = {}
        agents_on_switch_all = {}
        agents_near_to_switch = {}
        agents_near_to_switch_all = {}
        for a in range(self.env.get_num_agents()):
            ret_agents_on_switch, ret_agents_near_to_switch, ret_agents_near_to_switch_all, ret_agents_on_switch_all = \
                self.check_agent_decision(
                    self.env.agents[a].position,
                    self.env.agents[a].direction)
            agents_on_switch.update({a: ret_agents_on_switch})
            agents_on_switch_all.update({a: ret_agents_on_switch_all})
            ready_to_depart = self.env.agents[a].status == RailAgentStatus.READY_TO_DEPART
            agents_near_to_switch.update({a: (ret_agents_near_to_switch and not ready_to_depart)})

            agents_can_choose.update({a: agents_on_switch[a] or agents_near_to_switch[a]})

            agents_near_to_switch_all.update({a: (ret_agents_near_to_switch_all and not ready_to_depart)})

        return agents_can_choose, agents_on_switch, agents_near_to_switch, agents_near_to_switch_all, agents_on_switch_all

    def debug_render(self, env_renderer):
        agents_can_choose, agents_on_switch, agents_near_to_switch, agents_near_to_switch_all = \
            self.required_agent_decision()
        self.env.dev_obs_dict = {}
        for a in range(max(3, self.env.get_num_agents())):
            self.env.dev_obs_dict.update({a: []})

        selected_agent = None
        if agents_can_choose[0]:
            if self.env.agents[0].position is not None:
                self.debug_render_list.append(self.env.agents[0].position)
            else:
                self.debug_render_list.append(self.env.agents[0].initial_position)

        if self.env.agents[0].position is not None:
            self.debug_render_path_list.append(self.env.agents[0].position)
        else:
            self.debug_render_path_list.append(self.env.agents[0].initial_position)

        env_renderer.gl.agent_colors[0] = env_renderer.gl.rgb_s2i("FF0000")
        env_renderer.gl.agent_colors[1] = env_renderer.gl.rgb_s2i("666600")
        env_renderer.gl.agent_colors[2] = env_renderer.gl.rgb_s2i("006666")
        env_renderer.gl.agent_colors[3] = env_renderer.gl.rgb_s2i("550000")

        self.env.dev_obs_dict[0] = self.debug_render_list
        self.env.dev_obs_dict[1] = self.switches.keys()
        self.env.dev_obs_dict[2] = self.switches_neighbours.keys()
        self.env.dev_obs_dict[3] = self.debug_render_path_list

    def reset(self):
        self.build_data()
        return

    def fast_argmax(self, array):
        if array[0] == 1:
            return 0
        if array[1] == 1:
            return 1
        if array[2] == 1:
            return 2
        return 3

    def _explore(self, handle, new_position, new_direction, depth=0):
        has_opp_agent = 0
        has_same_agent = 0
        has_switch = 0
        visited = []

        # stop exploring (max_depth reached)
        if depth >= self.max_depth:
            return has_opp_agent, has_same_agent, has_switch, visited

        # max_explore_steps = 100
        cnt = 0
        while cnt < 100:
            cnt += 1

            visited.append(new_position)
            opp_a = self.env.agent_positions[new_position]
            if opp_a != -1 and opp_a != handle:
                if self.env.agents[opp_a].direction != new_direction:
                    # opp agent found
                    has_opp_agent = 1
                    return has_opp_agent, has_same_agent, has_switch, visited
                else:
                    has_same_agent = 1
                    return has_opp_agent, has_same_agent, has_switch, visited

            # convert one-hot encoding to 0,1,2,3
            agents_on_switch, \
            agents_near_to_switch, \
            agents_near_to_switch_all, \
            agents_on_switch_all = \
                self.check_agent_decision(new_position, new_direction)
            if agents_near_to_switch:
                return has_opp_agent, has_same_agent, has_switch, visited

            possible_transitions = self.env.rail.get_transitions(*new_position, new_direction)
            if agents_on_switch:
                f = 0
                for dir_loop in range(4):
                    if possible_transitions[dir_loop] == 1:
                        f += 1
                        hoa, hsa, hs, v = self._explore(handle,
                                                        get_new_position(new_position, dir_loop),
                                                        dir_loop,
                                                        depth + 1)
                        visited.append(v)
                        has_opp_agent += hoa
                        has_same_agent += hsa
                        has_switch += hs
                f = max(f, 1.0)
                return has_opp_agent / f, has_same_agent / f, has_switch / f, visited
            else:
                new_direction = fast_argmax(possible_transitions)
                new_position = get_new_position(new_position, new_direction)

        return has_opp_agent, has_same_agent, has_switch, visited

    def get(self, handle):
        # all values are [0,1]
        # observation[0]  : 1 path towards target (direction 0) / otherwise 0 -> path is longer or there is no path
        # observation[1]  : 1 path towards target (direction 1) / otherwise 0 -> path is longer or there is no path
        # observation[2]  : 1 path towards target (direction 2) / otherwise 0 -> path is longer or there is no path
        # observation[3]  : 1 path towards target (direction 3) / otherwise 0 -> path is longer or there is no path
        # observation[4]  : int(agent.status == RailAgentStatus.READY_TO_DEPART)
        # observation[5]  : int(agent.status == RailAgentStatus.ACTIVE)
        # observation[6]  : int(agent.status == RailAgentStatus.DONE or agent.status == RailAgentStatus.DONE_REMOVED)
        # observation[7]  : current agent is located at a switch, where it can take a routing decision
        # observation[8]  : current agent is located at a cell, where it has to take a stop-or-go decision
        # observation[9]  : current agent is located one step before/after a switch
        # observation[10] : 1 if there is a path (track/branch) otherwise 0 (direction 0)
        # observation[11] : 1 if there is a path (track/branch) otherwise 0 (direction 1)
        # observation[12] : 1 if there is a path (track/branch) otherwise 0 (direction 2)
        # observation[13] : 1 if there is a path (track/branch) otherwise 0 (direction 3)
        # observation[14] : If there is a path with step (direction 0) and there is a agent with opposite direction -> 1
        # observation[15] : If there is a path with step (direction 1) and there is a agent with opposite direction -> 1
        # observation[16] : If there is a path with step (direction 2) and there is a agent with opposite direction -> 1
        # observation[17] : If there is a path with step (direction 3) and there is a agent with opposite direction -> 1
        # observation[18] : If there is a path with step (direction 0) and there is a agent with same direction -> 1
        # observation[19] : If there is a path with step (direction 1) and there is a agent with same direction -> 1
        # observation[20] : If there is a path with step (direction 2) and there is a agent with same direction -> 1
        # observation[21] : If there is a path with step (direction 3) and there is a agent with same direction -> 1
        # observation[22] : If there is a switch on the path which agent can not use -> 1
        # observation[23] : If there is a switch on the path which agent can not use -> 1
        # observation[24] : If there is a switch on the path which agent can not use -> 1
        # observation[25] : If there is a switch on the path which agent can not use -> 1

        observation = np.zeros(self.observation_dim)
        visited = []
        agent = self.env.agents[handle]

        agent_done = False
        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_virtual_position = agent.initial_position
            observation[4] = 1
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_virtual_position = agent.position
            observation[5] = 1
        else:
            observation[6] = 1
            agent_virtual_position = (-1, -1)
            agent_done = True

        if not agent_done:
            visited.append(agent_virtual_position)
            distance_map = self.env.distance_map.get()
            current_cell_dist = distance_map[handle,
                                             agent_virtual_position[0], agent_virtual_position[1],
                                             agent.direction]
            possible_transitions = self.env.rail.get_transitions(*agent_virtual_position, agent.direction)
            orientation = agent.direction
            if fast_count_nonzero(possible_transitions) == 1:
                orientation = fast_argmax(possible_transitions)

            for dir_loop, branch_direction in enumerate([(orientation + dir_loop) % 4 for dir_loop in range(-1, 3)]):
                if possible_transitions[branch_direction]:
                    new_position = get_new_position(agent_virtual_position, branch_direction)
                    new_cell_dist = distance_map[handle,
                                                 new_position[0], new_position[1],
                                                 branch_direction]
                    if not (np.math.isinf(new_cell_dist) and np.math.isinf(current_cell_dist)):
                        observation[dir_loop] = int(new_cell_dist < current_cell_dist)

                    has_opp_agent, has_same_agent, has_switch, v = self._explore(handle, new_position, branch_direction)
                    visited.append(v)

                    observation[10 + dir_loop] = 1
                    observation[14 + dir_loop] = has_opp_agent
                    observation[18 + dir_loop] = has_same_agent
                    observation[22 + dir_loop] = has_switch

        agents_on_switch, \
        agents_near_to_switch, \
        agents_near_to_switch_all, \
        agents_on_switch_all = \
            self.check_agent_decision(agent_virtual_position, agent.direction)
        observation[7] = int(agents_on_switch)
        observation[8] = int(agents_near_to_switch)
        observation[9] = int(agents_near_to_switch_all)

        self.env.dev_obs_dict.update({handle: visited})

        return observation

    @staticmethod
    def agent_can_choose(observation):
        return observation[7] == 1 or observation[8] == 1
