import itertools
import os
import random
import time
import numpy as np
import warnings
from typing import List, Tuple, Union, Dict

from gComm.grammar import Grammar
from gComm.vocabulary import Vocabulary
from gComm.world import Object, World, ObjectVocabulary, EVENT, Position
from gComm.helpers import topo_sort


class gCommEnv(object):

    def __init__(self, intransitive_verbs: Union[Dict[str, str], List[str], int],
                 transitive_verbs: Union[Dict[str, str], List[str], int],
                 nouns: Union[Dict[str, str], List[str], int],
                 color_adjectives: Union[Dict[str, str], List[str], int],
                 size_adjectives: Union[Dict[str, str], List[str], int],
                 grid_size: int, min_object_size: int,
                 max_object_size: int, type_grammar: str,
                 maze_complexity: float, maze_density: float, enable_maze: bool,
                 save_directory=os.getcwd(), max_recursion=1, episode_len=20, wait=0.3,
                 lights_out=False, obstacles_flag=False, keep_fixed_weights=True, all_light=True):

        needed_type = list

        assert (isinstance(intransitive_verbs, needed_type) and isinstance(transitive_verbs, needed_type) and
                isinstance(nouns, needed_type) and isinstance(color_adjectives, needed_type) and
                isinstance(size_adjectives, needed_type))

        # All images, data and data statistics will be saved in this directory.
        self.save_directory = save_directory

        self.grid_size = grid_size

        # enable maze in Comm-gSCAN env
        self.enable_maze = enable_maze

        # Command vocabulary.
        self._vocabulary = Vocabulary.initialize(intransitive_verbs=intransitive_verbs,
                                                 transitive_verbs=transitive_verbs,
                                                 nouns=nouns,
                                                 color_adjectives=color_adjectives,
                                                 size_adjectives=size_adjectives)

        # Object vocabulary.
        self._object_vocabulary = ObjectVocabulary(shapes=self._vocabulary.get_semantic_shapes(),
                                                   colors=self._vocabulary.get_semantic_colors(),
                                                   min_size=min_object_size, max_size=max_object_size,
                                                   keep_fixed_weights=keep_fixed_weights,
                                                   all_light=all_light)

        # Initialize the world.
        self._world = World(grid_size=grid_size, colors=self._vocabulary.get_semantic_colors(),
                            object_vocabulary=self._object_vocabulary,
                            shapes=self._vocabulary.get_semantic_shapes(),
                            save_directory=self.save_directory,
                            episode_len=episode_len,
                            maze_complexity=maze_complexity,
                            maze_density=maze_density,
                            enable_maze=enable_maze,
                            lights_out=lights_out)

        # Generate the grammar.
        self._type_grammar = type_grammar
        self._grammar = Grammar(vocabulary=self._vocabulary, type_grammar=type_grammar, max_recursion=max_recursion)
        self._possible_situations = None
        self.obstacles_flag = obstacles_flag
        self.wait = wait  # wait time between consecutive time-step

        # [_ _ _ _    _      _       _       _    _ _ _ _   _   _ _ _ _]
        #  1 2 3 4 square cylinder circle diamond r b y g agent E S W N
        # _______ ______________________________ _______ _____ ________
        #   size                 shape            color  agent agent_dir
        self.grid_channels = 17
        self.grid_repr = None

        # [_ _ _ _   _       _       _       _    _ _ _ _   _     _    _    _    _     _   ]
        #  1 2 3 4 square cylinder circle diamond r b y g light heavy walk push pull pickup
        #  _______ ______________________________ _______ ___________ _____________________
        #    size               shape              color      weight           task
        self.concept_emb_dim = 18  # hard coded ..let's see!!!
        self.concept_repr = None

    def reset_env(self):
        self._world.clear_situation()
        self._world.carrying = None
        self._world.step_count = 0

    def generate_grid(self):
        self._world.gen_grid()

    @staticmethod
    def get_empty_situation():
        return {
            "target_shape": None,
            "target_color": None,
            "target_size": None,
            "target_position": None,
            "agent_position": None,
            "agent_direction": None}

    def generate_all_commands(self) -> {}:
        self._grammar.generate_all_commands()

    def generate_possible_targets(self, referred_size: str, referred_color: str, referred_shape: str):
        """
        Generate a list of possible target objects based on some target referred to in a command, e.g.
        for small red circle any sized circle but the largest can be a potential target.
        """
        if referred_size:
            if referred_size == "small":
                target_sizes = self._object_vocabulary.object_sizes[:-1]
            elif referred_size == "big":
                target_sizes = self._object_vocabulary.object_sizes[1:]
            else:
                raise ValueError("Unknown size adjective in command.")
        else:
            target_sizes = self._object_vocabulary.object_sizes
        # If no color specified, use all colors.
        if not referred_color:
            target_colors = self._object_vocabulary.object_colors
        else:
            target_colors = [referred_color]

        # Return all possible combinations of sizes and colors
        return list(itertools.product(target_sizes, target_colors, [referred_shape]))

    def get_larger_sizes(self, size: int) -> List[int]:
        return list(range(size + 1, self._object_vocabulary.largest_size + 1))

    def get_smaller_sizes(self, size: int) -> List[int]:
        return list(range(self._object_vocabulary.smallest_size, size))

    def generate_distinct_objects(self, referred_size: str, referred_color: str,
                                  referred_shape: str, actual_size: int, actual_color: str,
                                  sample_percentage: float, num_obstacles: int) -> Tuple[list, list, list]:
        """
        Generate a list of objects that are distinct from some referred target. E.g. if the referred target is a
        small circle, and the actual color of the target object is red, there cannot also be a blue circle of the same
        size, since then there will be 2 possible targets.
        Currently makes sure at least 2 sized objects of each group is placed whenever a size is referred to in the
        referred_size. E.g. if the command is 'walk to a big circle', make sure there are at least 2 sized circles.
        This doesn't get done for the color, e.g. if the comment is 'walk to a green circle', there are not
        necessarily also other colored circles in obligatory_objects.
        """
        objects = []
        # Initialize list that will be filled with objects that need to be present in the situation for it to make sense
        # E.g. if the referred object is 'small circle' there needs to be at least 1 larger circle.
        obligatory_objects = []
        # add obstacle_objects cells to the object list
        obstacle_objects = []
        if self.obstacles_flag is True:
            obstacle_objects.append(num_obstacles * ["wall"])
        # E.g. distinct from 'circle' -> no other circles; generate one random object of each other shape.
        if not referred_size and not referred_color:
            all_shapes = self._object_vocabulary.object_shapes
            all_shapes.remove(referred_shape)
            for shape in all_shapes:
                objects.append([(self._object_vocabulary.sample_size(), self._object_vocabulary.sample_color(), shape)])
            return objects, obligatory_objects, obstacle_objects
        # E.g. distinct from 'red circle' -> no other red circles of any size; generate one randomly size object for
        # each color, shape combination that is not a 'red circle'.
        elif not referred_size:
            for shape in self._object_vocabulary.object_shapes:
                for color in self._object_vocabulary.object_colors:
                    if color == referred_color and shape != referred_shape:
                        objects.append([(self._object_vocabulary.sample_size(), color, shape)])
                    elif color != referred_color and shape == referred_shape:
                        objects.append([(self._object_vocabulary.sample_size(), color, shape)])
                    elif not (shape == referred_shape and color == referred_color) and \
                            random.random() < sample_percentage:
                        objects.append([(self._object_vocabulary.sample_size(), color, shape)])
            return objects, obligatory_objects, obstacle_objects
        else:
            if referred_size == "small":
                all_other_sizes = self.get_larger_sizes(actual_size)
            elif referred_size == "big":
                all_other_sizes = self.get_smaller_sizes(actual_size)
            else:
                raise ValueError("Unknown referred size in command")
            all_other_shapes = self._object_vocabulary.object_shapes
            all_other_shapes.remove(referred_shape)
            # E.g. distinct from 'small circle' -> no circles of size <= than target in any color; generate two
            # random sizes for each color-shape pair except for the shape that is referred generate one larger objects
            # (if referred size is small, else a smaller object)
            if not referred_color:
                for shape in self._object_vocabulary.object_shapes:
                    for color in self._object_vocabulary.object_colors:
                        if not shape == referred_shape:
                            colored_shapes = []
                            for _ in range(2):
                                colored_shapes.append((self._object_vocabulary.sample_size(), color, shape))
                            objects.append(colored_shapes)
                        else:
                            if not color == actual_color:
                                colored_shapes = []
                                for _ in range(2):
                                    colored_shapes.append((random.choice(all_other_sizes), color, shape))
                                objects.append(colored_shapes)
                            else:
                                obligatory_objects.append((random.choice(all_other_sizes), color, shape))
                return objects, obligatory_objects, obstacle_objects
            # E.g. distinct from 'small red circle' -> no red circles of size <= as target; generate for each
            # color-shape pair two random sizes, and when the pair is the referred pair, one larger size.
            else:
                for shape in self._object_vocabulary.object_shapes:
                    for color in self._object_vocabulary.object_colors:
                        if not (shape == referred_shape and color == referred_color):
                            colored_shapes = []
                            for _ in range(2):
                                colored_shapes.append((self._object_vocabulary.sample_size(), color, shape))
                            objects.append(colored_shapes)
                        else:
                            obligatory_objects.append((random.choice(all_other_sizes), color, shape))
                return objects, obligatory_objects, obstacle_objects

    def generate_situation(self):
        """
        Generate a situation with an agent and a target object.
        :return: a dictionary with situations.
        """
        # All possible target objects
        all_targets = itertools.product(self._object_vocabulary.object_sizes, self._object_vocabulary.object_colors,
                                        self._object_vocabulary.object_shapes)

        # Loop over all semantically different situation specifications
        situation_specifications = {}
        largest_connected_reachable = 0  # keep track of the largest set of connected cells reachable from the agent
        for target_size, target_color, target_shape in all_targets:
            if target_shape not in situation_specifications.keys():
                situation_specifications[target_shape] = {}
            if target_color not in situation_specifications[target_shape].keys():
                situation_specifications[target_shape][target_color] = {}
            if target_size not in situation_specifications[target_shape][target_color].keys():
                situation_specifications[target_shape][target_color][target_size] = []

            empty_situation = self.get_empty_situation()
            agent_position = self._world.sample_position_conditioned()

            # find all reachable positions from the agent position (in presence of grid)
            self._world.obtain_reachable(agent_pos=agent_position)

            # situation where the agent is placed in some corner of the grid
            # with fewer reachable positions is unlikely
            if largest_connected_reachable > len(self._world.reachable) and random.random() > 0.6:
                continue
            else:
                largest_connected_reachable = max(largest_connected_reachable, len(self._world.reachable))

            # sample target positions
            target_position = self._world.get_position_at(agent_position)
            self._world.reachable.remove((target_position[0], target_position[1]))
            assert self._world.within_grid(target_position) and self._world.within_grid(agent_position)

            # Save a situation.
            empty_situation["agent_position"] = agent_position
            empty_situation["target_position"] = target_position
            empty_situation["target_shape"] = target_shape
            empty_situation["target_color"] = target_color
            empty_situation["target_size"] = target_size
            empty_situation["agent_direction"] = None
            situation_specifications[target_shape][target_color][target_size].append(empty_situation)

        return situation_specifications

    def initialize_world_from_spec(self, situation_spec, referred_size: str, referred_color: str,
                                   referred_shape: str, actual_size: int,
                                   sample_percentage=0.5, min_other_objects=0,
                                   max_other_objects=3, num_obstacles=0):
        # self._world.clear_situation()
        self._world.place_agent_at(situation_spec["agent_position"])

        target_shape = situation_spec["target_shape"]
        target_color = situation_spec["target_color"]
        target_size = situation_spec["target_size"]
        target_weight = self._object_vocabulary.object_in_class(target_size)

        self._world.place_object(Object(size=target_size, color=target_color, shape=target_shape,
                                        weight=target_weight),
                                 position=situation_spec["target_position"], target=True)

        distinct_objects, obligatory_objects, obstacle_objects = self.generate_distinct_objects(
            referred_size=self._vocabulary.translate_word(referred_size),
            referred_color=self._vocabulary.translate_word(referred_color),
            referred_shape=self._vocabulary.translate_word(referred_shape),
            actual_size=actual_size,
            actual_color=target_color,
            sample_percentage=sample_percentage,
            num_obstacles=num_obstacles)

        num_to_sample = len(distinct_objects)

        if min_other_objects > num_to_sample:
            warnings.warn("min_other_objects larger than sample population; dropping constraint")
        else:
            # num_to_sample = max(min_other_objects, num_to_sample)
            num_to_sample = max(min_other_objects, min(max_other_objects, len(distinct_objects)))

        sampled_objects = random.sample(distinct_objects, k=num_to_sample)

        objects_to_place = [obj[0] for obj in sampled_objects]
        random.shuffle(obligatory_objects)
        # max_other_objects = max_other_objects - len(sampled_objects)
        # NUM_OBJECTS_TO_ADD = max(0, min(max_other_objects, len(obligatory_objects)))
        # objects_to_place.extend(obligatory_objects[:NUM_OBJECTS_TO_ADD])
        objects_to_place.extend(obligatory_objects)

        # obtain all reachable positions
        self._world.obtain_reachable(Position(row=situation_spec["agent_position"][0],
                                              column=situation_spec["agent_position"][1]))

        for size, color, shape in objects_to_place:
            other_position = self._world.sample_position()

            # if all available positions are occupied or target is completely surrounded, skip the rest of the objects
            if other_position is None or \
                    self._world.is_target_surrounded(target_pos=situation_spec["target_position"],
                                                     sampled_object_pos=other_position):
                break

            weight = self._object_vocabulary.object_in_class(size)
            self._world.place_object(Object(size=size, color=color, shape=shape, weight=weight),
                                     position=other_position)

        if self.obstacles_flag is True:
            obstacles_to_place = [obj[0] for obj in obstacle_objects[0]]
            for _ in range(len(obstacles_to_place)):
                obstacle_position = self._world.sample_position()

                # if all available positions are occupied or target is completely surrounded, skip the obstacles
                if obstacle_position is None or \
                        self._world.is_target_surrounded(target_pos=situation_spec["target_position"],
                                                         sampled_object_pos=obstacle_position):
                    break

                self._world.place_object(Object(shape="wall", color="dimgrey", weight=None, size=None),
                                         position=obstacle_position, is_obstacle=True)

    def generate_world(self, other_objects_sample_percentage=0.5, max_other_objects=2,
                       min_other_objects=0, num_obstacles=0) -> {}:
        """
        Generate a set of situations and generate all possible commands based on the current grammar and lexicon,
        match commands to situations based on relevance (if a command refers to a target object, it needs to be
        present in the situation) and save these pairs in a the list of data examples.
        """

        # reset episode
        self.reset_env()

        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode
        self.generate_grid()

        # Generate all situations and commands.
        self._grammar.reset_grammar()
        self._possible_situations = self.generate_situation()

        self.generate_all_commands()
        template_num, template_derivations = random.sample(list(self._grammar.all_derivations.items()), 1)[0]
        # template derivations is a list
        random.shuffle(template_derivations)
        derivation = random.sample(template_derivations, 1)[0]

        arguments = []
        derivation.meaning(arguments)
        assert len(arguments) == 1, "Only one target object currently supported."

        target_str, target_predicate = arguments.pop().to_predicate()
        possible_target_objects = self.generate_possible_targets(
            referred_size=self._vocabulary.translate_word(target_predicate["size"]),
            referred_color=self._vocabulary.translate_word(target_predicate["color"]),
            referred_shape=self._vocabulary.translate_word(target_predicate["noun"]))

        # if self._possible_situations is None:
        random.shuffle(possible_target_objects)
        while True:
            sampled_target = random.sample(possible_target_objects, k=1)
            target_size, target_color, target_shape = sampled_target[0]

            try:
                assert len(self._possible_situations[target_shape][target_color][target_size]) >= 1
                relevant_situation = self._possible_situations[target_shape][target_color][target_size][0]
                self.initialize_world_from_spec(relevant_situation,
                                                referred_size=target_predicate["size"],
                                                referred_color=target_predicate["color"],
                                                referred_shape=target_predicate["noun"],
                                                actual_size=target_size,
                                                sample_percentage=other_objects_sample_percentage,
                                                min_other_objects=min_other_objects,
                                                max_other_objects=max_other_objects,
                                                num_obstacles=num_obstacles)

                # self._world.get_current_situation()
                verb_in_command = self.extract_verb(derivation)
                self._world.target_verb = verb_in_command
                return ' '.join(derivation.words()), verb_in_command
            except AssertionError:
                # print('Assertion error')
                pass

    @staticmethod
    def extract_verb(derivation):
        arguments = []
        logical_form = derivation.meaning(arguments)

        # Extract all present events in the current command and order them by constraints.
        events = [variable for variable in logical_form.variables if variable.sem_type == EVENT]
        seq_constraints = [term.arguments for term in logical_form.terms if term.function == "seq"]
        ordered_events = topo_sort(events, seq_constraints)

        # Loop over the events to get the demonstrations.
        action = None
        for event in ordered_events:
            # Get the logical form of the current event
            sub_logical_form = logical_form.select([event], exclude={"seq"})
            event_lf = sub_logical_form.select([event], exclude={"patient"})

            # Find the action verb if it exists.
            if event_lf.head.sem_type == EVENT:
                for term in event_lf.terms:
                    if term.specs.action:
                        action = term.specs.action
        return action

    def grid_input(self, specify_target=False) -> np.ndarray:
        """
        Each grid cell in an episode is fully specified by a vector:
        [_ _ _ _    _      _       _       _    _ _ _ _   _   _ _ _ _]
         1 2 3 4 square cylinder circle diamond r b y g agent E S W N
         _______ ______________________________ _______ _____ ________
           size                 shape            color  agent agent_dir
        :return: grid representation
        """
        self.grid_repr = np.zeros([self.grid_size, self.grid_size, self.grid_channels], dtype=int)
        situation = self._world.get_current_situation()

        # self.grid_repr[situation.agent_pos.row, situation.agent_pos.column, -5] = 1  # set True if agent
        # self.grid_repr[situation.agent_pos.row, situation.agent_pos.column, -4 + self._world.agent_dir] = 1

        target = situation.target_object
        self.grid_repr[target.position.row, target.position.column, :] = \
            np.concatenate([target.vector, np.zeros([self.grid_channels - len(target.vector)], dtype=np.int)])

        for distractor in situation.placed_objects:
            self.grid_repr[distractor.position.row, distractor.position.column, :] = \
                np.concatenate([distractor.vector,
                                np.zeros([self.grid_channels - len(distractor.vector)], dtype=np.int)])

        if self.enable_maze:
            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    if self._world.maze[row, col] == 1:
                        self.grid_repr[row, col, :] = np.ones(self.grid_channels)

        # fill agent info
        self.grid_repr[situation.agent_pos.row, situation.agent_pos.column, -5] = 1  # set True if agent
        self.grid_repr[situation.agent_pos.row, situation.agent_pos.column, -4 + self._world.agent_dir] = 1

        if specify_target is True:
            pad = np.zeros([self.grid_size, self.grid_size], dtype=int)
            pad[target.position.row, target.position.column] = 1
            self.grid_repr = np.dstack((self.grid_repr, pad))

        return self.grid_repr

    def concept_input(self, task_verb: str) -> [np.ndarray, str]:
        """
        The target object and the task is fully specified by a vector:
        [_ _ _ _   _       _       _       _    _ _ _ _   _     _    _    _    _     _   ]
         1 2 3 4 square cylinder circle diamond r b y g light heavy walk push pull pickup
         _______ ______________________________ _______ ___________ _____________________
           size               shape              color      weight           task
        """
        self.concept_repr = np.zeros(self.concept_emb_dim)
        situation = self._world.get_current_situation()
        target_vector = situation.target_object.vector
        self.concept_repr[:len(target_vector)] = target_vector
        self.concept_repr[len(target_vector): len(target_vector) + 2] = \
            self._world.weight_representation(situation.target_object[0].weight)
        self.concept_repr[-4:] = self._world.task_representation(task_verb)

        target_weight = situation.target_object[0].weight
        return self.concept_repr, target_weight

    def grid_image_input(self):
        return self._world.render(mode="human", image_input=True).getArray()

    def render_episode(self, mission='', countdown='', actions='', weight='light',
                       verb_in_command='', save_fig=False, save_path=None):

        if verb_in_command in ['push', 'pull']:
            self._world.render_situation(mission=mission,
                                         countdown=countdown,
                                         actions=actions,
                                         weight=weight,
                                         save_fig=save_fig,
                                         save_path=save_path)
        else:
            self._world.render_situation(mission=mission,
                                         countdown=countdown,
                                         actions=actions,
                                         save_fig=save_fig,
                                         save_path=save_path)

        time.sleep(self.wait)

    def step(self, action):
        return self._world.step(action)
