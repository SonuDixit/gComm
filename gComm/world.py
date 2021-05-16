import itertools
import os
import random
import warnings
from collections import namedtuple
from itertools import product
from typing import Tuple, List, Dict

import matplotlib.pyplot as plt
import numpy as np

from gComm.helpers import one_hot, generate_possible_object_names
from gComm.mazegrid_base.mazegrid import Grid, IDX_TO_OBJECT, OBJECT_TO_IDX, DIR_TO_VEC, \
    Circle, Square, Cylinder, Diamond, Wall
from gComm.mazegrid_base.mazegrid import MiniGridEnv

SemType = namedtuple("SemType", "name")
Position = namedtuple("Position", "row column")
Object = namedtuple("Object", "size color shape weight")
PositionedObject = namedtuple("PositionedObject", "object position vector")  # , defaults=(None, None, None)
PositionedObject.__new__.__defaults__ = (None,) * len(PositionedObject._fields)
Variable = namedtuple("Variable", "name sem_type")
fields = ("action", "is_transitive", "adjective_type", "noun")
Weights = namedtuple("Weights", fields)  # , defaults=(None,) * len(fields)
Weights.__new__.__defaults__ = (None,) * len(fields)

ENTITY = SemType("noun")
COLOR = SemType("color")
SIZE = SemType("size")
EVENT = SemType("verb")

Direction = namedtuple("Direction", "name")
NORTH = Direction("north")
SOUTH = Direction("south")
WEST = Direction("west")
EAST = Direction("east")
FORWARD = Direction("forward")

DIR_TO_INT = {
    NORTH: 3,
    SOUTH: 1,
    WEST: 2,
    EAST: 0
}

INT_TO_DIR = {direction_int: direction for direction, direction_int in DIR_TO_INT.items()}

SIZE_TO_INT = {
    "small": 1,
    "average": 2,
    "big": 3
}

DIR_STR_TO_DIR = {
    "n": NORTH,
    "e": EAST,
    "s": SOUTH,
    "w": WEST,
}
# "null": agent_pos = target_pos
DIR_VEC_TO_DIR = {
    (0, 0): "null",
    (1, 0): "e",
    (0, 1): "n",
    (-1, 0): "w",
    (0, -1): "s",
    (1, 1): "ne",
    (1, -1): "se",
    (-1, -1): "sw",
    (-1, 1): "nw"
}

weight_repr = {
    'light': [1, 0],
    'heavy': [0, 1]
}

task_repr = {
    'walk': [1, 0, 0, 0],
    'push': [0, 1, 0, 0],
    'pull': [0, 0, 1, 0],
    'pickup': [0, 0, 0, 1]
}

Command = namedtuple("Command", "action event")
UNK_TOKEN = 'UNK'


class Term(object):
    """
    Holds terms that can be parts of logical forms and take as arguments variables that the term can operate over.
    E.g. for the phrase 'Brutus stabs Caesar' the term is stab(B, C) which will be represented by the string
    "(stab B:noun C:noun)".
    """

    def __init__(self, function: str, args: tuple, weights=None, meta=None, specs=None):
        self.function = function
        self.arguments = args
        self.weights = weights
        self.meta = meta
        self.specs = specs

    def replace(self, var_to_find: Variable, replace_by_var: Variable):
        """Find a variable `var_to_find` the arguments and replace it by `replace_by_var`."""
        return Term(
            function=self.function,
            args=tuple(replace_by_var if variable == var_to_find else variable for variable in self.arguments),
            specs=self.specs,
            meta=self.meta
        )

    def to_predicate(self, predicate: dict):
        assert self.specs is not None
        output = self.function
        if self.specs.noun:
            predicate["noun"] = output
        elif self.specs.adjective_type == SIZE:
            predicate["size"] = output
        elif self.specs.adjective_type == COLOR:
            predicate["color"] = output

    def __repr__(self):
        parts = [self.function]
        for variable in self.arguments:
            parts.append("{}:{}".format(variable.name, variable.sem_type.name))
        return "({})".format(" ".join(parts))


class LogicalForm(object):
    """
    Holds neo-Davidsonian-like logical forms (http://ling.umd.edu//~alxndrw/LectureNotes07/neodavidson_intro07.pdf).
    An object LogicalForm(variables=[x, y, z], terms=[t1, t2]) may represent
    lambda x, y, z: and(t1(x, y, z), t2(x, y, z)) (depending on which terms involve what variables).
    """

    def __init__(self, variables: Tuple[Variable], terms: Tuple[Term]):
        self.variables = variables
        self.terms = terms
        if len(variables) > 0:
            self.head = variables[0]

    def bind(self, bind_var: Variable):
        """
        Bind a variable to its head, e.g for 'kick the ball', 'kick' is the head and 'the ball' will be bind to it.
        Or in the case of NP -> JJ NP, bind the JJ (adjective) to the head of the noun-phrase.
        E.g. 'the big red square', bind 'big' to 'square'.
        :param bind_var:
        :return:
        """
        sub_var, variables_out = self.variables[0], self.variables[1:]
        terms_out = [term.replace(sub_var, bind_var) for term in self.terms]
        return LogicalForm(variables=(bind_var,) + variables_out, terms=tuple(terms_out))

    def select(self, variables: list, exclude=frozenset()):
        """Select and return the sub-logical form of the variables in the variables list."""
        queue = list(variables)
        used_vars = set()
        terms_out = []
        while len(queue) > 0:
            var = queue.pop()
            deps = [term for term in self.terms if term.function not in exclude and term.arguments[0] == var]
            for term in deps:
                terms_out.append(term)
                used_vars.add(var)
                for variable in term.arguments[1:]:
                    if variable not in used_vars:
                        queue.append(variable)

        vars_out = [var for var in self.variables if var in used_vars]
        terms_out = list(set(terms_out))
        return LogicalForm(tuple(vars_out), tuple(terms_out))

    def to_predicate(self):
        assert len(self.variables) == 1
        predicate = {"noun": "", "size": "", "color": ""}
        [term.to_predicate(predicate) for term in self.terms]
        object_str = ""
        if predicate["color"]:
            object_str += ' ' + predicate["color"]
        object_str += ' ' + predicate["noun"]
        object_str = object_str.strip()
        return object_str, predicate

    def __repr__(self):
        return "LF({})".format(" ^ ".join([repr(term) for term in self.terms]))


def object_to_repr(object: Object) -> dict:
    return {
        "shape": object.shape,
        "color": object.color,
        "size": str(object.size),
        "weight": object.weight
    }


def position_to_repr(position: Position) -> dict:
    return {
        "row": str(position.row),
        "column": str(position.column)
    }


def positioned_object_to_repr(positioned_object: PositionedObject) -> dict:
    return {
        "vector": ''.join([str(idx) for idx in positioned_object.vector]),
        "position": position_to_repr(positioned_object.position),
        "object": object_to_repr(positioned_object.object)
    }


def parse_object_repr(object_repr: dict) -> Object:
    return Object(shape=object_repr["shape"], color=object_repr["color"],
                  size=int(object_repr["size"]), weight=object_repr["weight"])


def parse_position_repr(position_repr: dict) -> Position:
    return Position(row=int(position_repr["row"]), column=int(position_repr["column"]))


def parse_object_vector_repr(object_vector_repr: str) -> np.ndarray:
    return np.array([int(idx) for idx in object_vector_repr])


def parse_positioned_object_repr(positioned_object_repr: dict):
    return PositionedObject(object=parse_object_repr(positioned_object_repr["object"]),
                            position=parse_position_repr(positioned_object_repr["position"]),
                            vector=parse_object_vector_repr(positioned_object_repr["vector"]))


class Situation(object):
    """
    Specification of a situation that can be used for serialization as well as initialization of a world state.
    """

    def __init__(self, grid_size: int, agent_position: Position, agent_direction: Direction,
                 target_object: PositionedObject, placed_objects: List[PositionedObject]):
        self.grid_size = grid_size
        self.agent_pos = agent_position  # position is [col, row] (i.e. [x-axis, y-axis])
        self.agent_direction = agent_direction
        self.placed_objects = placed_objects
        self.target_object = target_object

    @property
    def distance_to_target(self):
        """Number of grid steps to take to reach the target position from the agent position."""
        return abs(self.agent_pos.column - self.target_object.position.column) + \
               abs(self.agent_pos.row - self.target_object.position.row)

    @property
    def direction_to_target(self):
        """Direction to the target in terms of north, east, south, north-east, etc. Needed for a grounded scan split."""
        column_distance = self.target_object.position.column - self.agent_pos.column
        column_distance = min(max(-1, column_distance), 1)
        row_distance = self.agent_pos.row - self.target_object.position.row
        row_distance = min(max(-1, row_distance), 1)
        return DIR_VEC_TO_DIR[(column_distance, row_distance)]

    def to_dict(self) -> dict:
        """Represent this situation in a dictionary."""
        return {
            "agent_position": Position(row=self.agent_pos[0], column=self.agent_pos[1]),
            "agent_direction": self.agent_direction,
            "target_object": self.target_object,
            "grid_size": self.grid_size,
            "objects": self.placed_objects
        }

    def to_representation(self) -> dict:
        """Represent this situation in serializable dict that can be written to a file."""
        return {
            "grid_size": self.grid_size,
            "agent_position": position_to_repr(self.agent_pos),
            "agent_direction": DIR_TO_INT[self.agent_direction],
            "target_object": positioned_object_to_repr(self.target_object) if self.target_object else None,
            "distance_to_target": str(self.distance_to_target) if self.target_object else None,
            "direction_to_target": self.direction_to_target if self.target_object else None,
            "placed_objects": {str(i): positioned_object_to_repr(placed_object) for i, placed_object
                               in enumerate(self.placed_objects)}
        }

    @classmethod
    def from_representation(cls, situation_representation: dict):
        """Initialize this class by some situation as represented by .to_representation()."""
        target_object = situation_representation["target_object"]
        placed_object_reps = situation_representation["placed_objects"]
        placed_objects = []
        for placed_object_rep in placed_object_reps.values():
            placed_objects.append(parse_positioned_object_repr(placed_object_rep))
        situation = cls(grid_size=situation_representation["grid_size"],
                        agent_position=parse_position_repr(situation_representation["agent_position"]),
                        agent_direction=INT_TO_DIR[situation_representation["agent_direction"]],
                        target_object=parse_positioned_object_repr(target_object) if target_object else None,
                        placed_objects=placed_objects)
        return situation


class ObjectVocabulary(object):
    """
    Constructs an object vocabulary. Each object will be calculated by the following:
    [size color shape] and where size is on an ordinal scale of 1 (smallest) to 4 (largest),
    colors and shapes are orthogonal vectors [0 1] and [1 0] and the result is a concatenation:
    e.g. the biggest red circle: [4 0 1 0 1], the smallest blue square: [1 1 0 1 0]
    """
    SIZES = list(range(1, 5))

    def __init__(self, shapes: List[str], colors: List[str], min_size: int, max_size: int,
                 keep_fixed_weights: bool, all_light: bool):
        """
        :param shapes: a list of string names for nouns.
        :param colors: a list of string names for colors.
        :param min_size: minimum object size
        :param max_size: maximum object size
        """
        assert self.SIZES[0] <= min_size <= max_size <= self.SIZES[-1], \
            "Unsupported object sizes (min: {}, max: {}) specified.".format(min_size, max_size)
        self._min_size = min_size
        self._max_size = max_size

        # Translation from shape nouns to shapes.
        self._shapes = set(shapes)
        self._n_shapes = len(self._shapes)
        self._colors = set(colors)
        self._n_colors = len(self._colors)
        self._idx_to_shapes_and_colors = shapes + colors
        self._shapes_and_colors_to_idx = {token: i for i, token in enumerate(self._idx_to_shapes_and_colors)}
        self._sizes = list(range(min_size, max_size + 1))

        # Also size specification for 'average' size, e.g. if adjectives are small and big, 3 sizes exist.
        self._n_sizes = len(self._sizes)
        assert (self._n_sizes % 2) == 0, "Please specify an even amount of sizes " \
                                         " (needs to be split in 2 classes.)"

        if all_light:  # fix all light weights
            light = [1, 2, 3, 4]
            heavy = []
        else:
            if keep_fixed_weights is True:  # fix weights according to size (smaller size is lighter and vice versa)
                light = [1, 2]
                heavy = [3, 4]
            else:  # fix weights randomly
                light = np.random.choice(np.arange(1, 5), 2, replace=False)
                heavy = [i for i in range(1, 5) if i not in light]

        # Make object classes.
        self._object_class = {i: "light" for i in light}
        self._heavy_weights = {i: "heavy" for i in heavy}
        self._object_class.update(self._heavy_weights)

        # Prepare object vectors.
        self._object_vector_size = self._n_shapes + self._n_colors + self._n_sizes
        self._object_vectors = self.generate_objects()
        self._possible_colored_objects = set([color + ' ' + shape for color, shape in itertools.product(self._colors,
                                                                                                        self._shapes)])

    def has_object(self, shape: str, color: str, size: int):
        return shape in self._shapes and color in self._colors and size in self._sizes

    def object_in_class(self, size: int):
        return self._object_class[size]

    @property
    def num_object_attributes(self):
        """Dimension of object vectors is one hot for shapes and colors + 1 ordinal dimension for size."""
        return len(self._idx_to_shapes_and_colors) + self._n_sizes

    @property
    def smallest_size(self):
        return self._min_size

    @property
    def largest_size(self):
        return self._max_size

    @property
    def object_shapes(self):
        return self._shapes.copy()

    @property
    def object_sizes(self):
        return self._sizes.copy()

    @property
    def object_colors(self):
        return self._colors.copy()

    @property
    def all_objects(self):
        return product(self.object_sizes, self.object_colors, self.object_shapes)

    def sample_size(self):
        return random.choice(self._sizes)

    def sample_color(self):
        return random.choice(list(self._colors))

    def get_object_vector(self, shape: str, color: str, size: int):
        # assert self.has_object(shape, color, size), "Trying to get an unavailable object vector from the vocabulary/"
        if shape == "wall":
            return self._object_vectors["wall"]
        return self._object_vectors[shape][color][size]

    def generate_objects(self) -> Dict[str, Dict[str, Dict[int, np.ndarray]]]:
        """
        An object vector is built as follows: the first entry is an ordinal entry defining the size (from 1 the
        smallest to 4 the largest), then 2 entries define a one-hot vector over shape, the last two entries define a
        one-hot vector over color. A red circle of size 1 could then be: [1 0 1 0 1], a blue square of size 2 would
        be [2 1 0 1 0].
        """
        object_vector = None
        object_to_object_vector = {}
        for size, color, shape in itertools.product(self._sizes, self._colors, self._shapes):
            object_vector = one_hot(self._object_vector_size, size - 1) + \
                            one_hot(self._object_vector_size, self._shapes_and_colors_to_idx[shape] + self._n_sizes) + \
                            one_hot(self._object_vector_size, self._shapes_and_colors_to_idx[color] + self._n_sizes)
            # object_vector = np.concatenate(([size], object_vector))
            if shape not in object_to_object_vector.keys():
                object_to_object_vector[shape] = {}
            if color not in object_to_object_vector[shape].keys():
                object_to_object_vector[shape][color] = {}
            object_to_object_vector[shape][color][size] = object_vector

        # hardcoded wall vector
        object_to_object_vector["wall"] = np.ones_like(object_vector)

        return object_to_object_vector


class World(MiniGridEnv):
    """
    Wrapper class to execute actions in a world state. Connected to minigrid.py in gym_minigrid for visualizations.
    Every time actions are executed, the commands and situations are saved in self._observed_commands and
    self._observed_situations, which can then be retrieved with get_current_observations().
    The world can be cleared with clear_situation().
    """

    AVAILABLE_SHAPES = {"circle", "square", "cylinder", "diamond", "wall"}
    AVAILABLE_COLORS = {"red", "blue", "green", "yellow", "dimgrey"}

    def __init__(self, grid_size: int, shapes: List[str], colors: List[str], object_vocabulary: ObjectVocabulary,
                 save_directory: str, episode_len: int, maze_complexity: float, maze_density: float, enable_maze: bool,
                 lights_out=False):
        # Some checks on the input
        for shape, color in zip(shapes, colors):
            assert shape in self.AVAILABLE_SHAPES, "Specified shape {} not implemented in minigrid env.".format(shape)
            assert color in self.AVAILABLE_COLORS, "Specified color {}, not implemented in minigrid env.".format(color)

        # Define the grid world.
        self.grid_size = grid_size

        # Column, row
        self.agent_start_pos = None
        self.agent_start_dir = None
        self.mission = None

        # Generate the object vocabulary.
        self._object_vocabulary = object_vocabulary
        self.num_available_objects = len(IDX_TO_OBJECT.keys())
        self.available_objects = set(OBJECT_TO_IDX.keys())

        # Data structures for keeping track of the current state of the world.
        self._placed_object_list = []
        self._target_object = None
        self.target_verb = None
        self._observed_commands = []
        self._observed_situations = []
        self._occupied_positions = set()
        self._num_wall_blocks = 0
        # Hash table for looking up locations of objects based on partially formed references (e.g. find the location(s)
        # of a red cylinder when the grid has both a big red cylinder and a small red cylinder.)
        self._object_lookup_table = {}
        self.save_directory = save_directory

        # reward based attributes
        self.episode_len = episode_len

        # lights off feature
        self.lights_out = lights_out

        # reachable
        self.reachable = set()

        # maze generation
        self.maze_complexity = maze_complexity
        self.maze_density = maze_density
        self.enable_maze = enable_maze
        super().__init__(grid_size=grid_size, max_steps=4 * grid_size * grid_size)

    def gen_grid(self):
        # Create an empty grid
        self.grid = Grid(width=self.grid_size, height=self.grid_size,
                         depth=self._object_vocabulary.num_object_attributes, lights_out=self.lights_out)

        # generate maze
        if self.enable_maze is True:
            assert self.grid_size % 2 == 0, "grid_size must be even when enable_maze is True"
            self.maze = self.generate_maze(complexity=self.maze_complexity, density=self.maze_density)
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if self.maze[i][j] == 1:
                        self.place_object(Object(shape="wall", color="dimgrey", weight=None, size=None),
                                          position=Position(row=i, column=j), is_obstacle=True)
                        self._num_wall_blocks += 1

    @staticmethod
    def create_object(object_spec: Object, object_vector: np.ndarray, target=False, lights_out=False):
        if object_spec.shape == "circle":
            return Circle(object_spec.color, size=object_spec.size, vector_representation=object_vector,
                          object_representation=object_spec, target=target,
                          weight=object_spec.weight,
                          lights_out=lights_out)
        elif object_spec.shape == "square":
            return Square(object_spec.color, size=object_spec.size, vector_representation=object_vector,
                          object_representation=object_spec, target=target,
                          weight=object_spec.weight,
                          lights_out=lights_out)
        elif object_spec.shape == "cylinder":
            return Cylinder(object_spec.color, size=object_spec.size, vector_representation=object_vector,
                            object_representation=object_spec,
                            weight=object_spec.weight,
                            lights_out=lights_out)
        elif object_spec.shape == "diamond":
            return Diamond(object_spec.color, size=object_spec.size, vector_representation=object_vector,
                           object_representation=object_spec,
                           weight=object_spec.weight,
                           lights_out=lights_out)
        elif object_spec.shape == "wall":
            return Wall(vector_representation=object_vector, object_representation=object_spec, lights_out=lights_out)
        else:
            raise ValueError("Trying to create an object shape {} that is not implemented.".format(object_spec.shape))

    def position_taken(self, position: Position):
        return self.grid.get(position.row, position.column) is not None

    def within_grid(self, position: Position) -> bool:
        if 0 <= position.row < self.grid_size and 0 <= position.column < self.grid_size:
            return True
        else:
            return False

    def place_agent_at(self, position: Position):
        if not self.position_taken(position):
            self.place_agent(top=(position.row, position.column), size=(1, 1), rand_dir=True)
            self._occupied_positions.add((position.row, position.column))
        else:
            raise ValueError("Trying to place agent on cell that is already taken.")

    def sample_position(self):
        available_positions = [(row, col) for row, col in self.reachable if (row, col) not in self._occupied_positions]
        if not available_positions:
            warnings.warn("available positions full; discarding remaining objects")
            return None
        sampled_position = random.sample(available_positions, 1)[0]
        self.reachable.remove((sampled_position[0], sampled_position[1]))
        return Position(row=sampled_position[0], column=sampled_position[1])

    def sample_position_conditioned(self) -> Position:
        available_positions = [(row, col) for row, col in itertools.product(list(range(self.grid_size)),
                                                                            list(range(self.grid_size)))
                               if (row, col) not in self._occupied_positions]
        sampled_position = random.sample(available_positions, 1).pop()
        return Position(row=sampled_position[0], column=sampled_position[1])

    def place_object(self, object_spec: Object, position: Position, target=False, is_obstacle=False):
        object_vector = None
        if not self.within_grid(position):
            if is_obstacle:
                raise IndexError("Trying to place obstacle outside of grid of size {}.".format(self.grid_size))
            else:
                raise IndexError("Trying to place object '{}' outside of grid of size {}.".format(
                    object_spec.shape, self.grid_size))
        # Object already placed at this location
        if self.position_taken(position):
            warnings.warn("attempting to place two objects at location ({}, {}), but overlapping objects not "
                          "supported. Skipping object. \n".format(position.row, position.column))
        else:
            object_vector = self._object_vocabulary.get_object_vector(shape=object_spec.shape,
                                                                      color=object_spec.color,
                                                                      size=object_spec.size)

            positioned_object = PositionedObject(object=object_spec, position=position, vector=object_vector)
            self.place_obj(self.create_object(object_spec, object_vector, target=target, lights_out=self.lights_out),
                           top=(position.row, position.column), size=(1, 1))

            # Add to list that keeps track of all objects currently positioned on the grid.
            self._placed_object_list.append(positioned_object)

            # Adjust the object lookup table accordingly.
            self._add_object_to_lookup_table(positioned_object)

            # Add to occupied positions:
            self._occupied_positions.add((position.row, position.column))

            if target:
                self._target_object = positioned_object
        return object_vector

    def is_target_surrounded(self, target_pos: Position, sampled_object_pos: Position) -> bool:
        """
        preventing all 4 positions around the target object
        from being occupied. For push and pull tasks
        :return: False if target is not completely surrounded (more than 1 free cell);
                 else True
        """
        rowNbr = [-1, 0, 0, 1]
        colNbr = [0, -1, 1, 0]

        # all adjoining cells of the target_object
        all_adjoining = \
            [Position(row=target_pos.row + rowNbr[k], column=target_pos.column + colNbr[k]) for k in range(4)]

        # check if sampled object position is in a adjoining cell
        if sampled_object_pos not in all_adjoining:
            return False

        # check all adjoining cells of the target to see if they are occupied (by other objects or the agent)
        free = 4  # count of free positions around the target object
        for pos_cell in all_adjoining:
            if self.within_grid(pos_cell):
                if self.grid.get(pos_cell.row, pos_cell.column) or \
                        (pos_cell.row, pos_cell.column) == tuple(self.agent_pos):
                    free -= 1
            else:
                free -= 1

        if free > 1:  # more than 1 free cell, hence return False
            return False

        warnings.warn('Target objects surrounded on 3 sides; Discarding rest of the objects')
        return True  # 1 or no free cells, hence return True

    def all_reachable_positions(self, agent_pos) -> set:
        """
        Check if target object is reachable in the presence of obstacles and maze.
        Initially, the maze is created and the obstacles are placed depending on
        the difficulty level (degree of complexity).
        The agent position is sampled. The target and distractor positions
        are subsequently sampled from the list of reachable positions
        """

        def DFS(row, column):
            rowNbr = [-1, 0, 0, 1]
            colNbr = [0, -1, 1, 0]

            # Recur for all connected neighbours
            for k in range(4):
                pos_cell = (row + rowNbr[k], column + colNbr[k])
                if (0 <= pos_cell[0] < self.grid_size) \
                        and (0 <= pos_cell[1] < self.grid_size) \
                        and (pos_cell not in reachable) \
                        and (self.maze[pos_cell[0], pos_cell[1]] == 0):
                    # Add this cell to reachable
                    reachable.add((pos_cell[0], pos_cell[1]))
                    DFS(pos_cell[0], pos_cell[1])

        # Reachable positions
        reachable = set()
        DFS(row=agent_pos[0], column=agent_pos[1])
        return reachable

    def obtain_reachable(self, agent_pos: Position):
        if self.enable_maze is True:
            self.reachable = self.all_reachable_positions(agent_pos=agent_pos)
        else:
            self.reachable = set()
            for row in range(self.height):
                for col in range(self.width):
                    self.reachable.add((row, col))
            self.reachable.remove(agent_pos)

    def _add_object_to_lookup_table(self, positioned_object: PositionedObject):
        object_size = positioned_object.object.size
        object_color = positioned_object.object.color
        object_shape = positioned_object.object.shape

        # Generate all possible names
        object_names = generate_possible_object_names(color=object_color, shape=object_shape)
        for possible_object_name in object_names:
            if possible_object_name not in self._object_lookup_table.keys():
                self._object_lookup_table[possible_object_name] = {}

            # This part allows for multiple exactly the same objects (e.g. 2 small red circles) to be on the grid.
            if object_size is not None:
                if positioned_object.object.size not in self._object_lookup_table[possible_object_name].keys():
                    self._object_lookup_table[possible_object_name] = {
                        size: [] for size in self._object_vocabulary.object_sizes}
                self._object_lookup_table[possible_object_name][object_size].append(
                    positioned_object.position)

    def _remove_object_from_lookup_table(self, positioned_object: PositionedObject):
        possible_object_names = generate_possible_object_names(positioned_object.object.color,
                                                               positioned_object.object.shape)
        for possible_object_name in possible_object_names:
            self._object_lookup_table[possible_object_name][positioned_object.object.size].remove(
                positioned_object.position)

    def _remove_object(self, from_position: Position) -> PositionedObject:
        # remove from placed_object_list
        target_object = None

        for i, positioned_object in enumerate(self._placed_object_list):
            if positioned_object.position == from_position:
                target_object = self._placed_object_list[i]
                del self._placed_object_list[i]
                break

        # remove from object_lookup Table
        self._remove_object_from_lookup_table(target_object)

        # remove from gym grid
        self.grid.get(from_position.row, from_position.column)
        self.grid.set(from_position.row, from_position.column, None)

        self._occupied_positions.remove((from_position.row, from_position.column))
        return target_object

    def move_object(self, old_position: Position, new_position: Position, target=False):
        # Remove object from old position
        old_positioned_object = self._remove_object(from_position=old_position)
        if not old_positioned_object:
            raise ValueError("Trying to move an object from an empty grid location (row {}, col {})".format(
                old_position.row, old_position.column))

        # Add object at new position
        self.place_object(old_positioned_object.object, new_position, target)

    @staticmethod
    def get_direction(direction_str: str):
        return DIR_STR_TO_DIR[direction_str]

    @staticmethod
    def weight_representation(weight) -> list:
        return weight_repr[weight]

    @staticmethod
    def task_representation(task) -> list:
        return task_repr[task]

    @property
    def placed_object_list(self):
        return self._placed_object_list

    @property
    def num_wall_blocks(self):
        return self._num_wall_blocks

    def get_position_at(self, agent_position: Position) -> Position:
        """
        Find the farthest positions from the target position
        sample a position amongst them.
        Distance metric used: Manhattan distance
        """

        def manhattan_dist(xA, yA):
            return abs(xA - agent_position[0]) + abs(yA - agent_position[1])

        dist_zipped = list(zip(list(self.reachable), [manhattan_dist(xA, yA) for (xA, yA) in self.reachable]))
        dist_sorted = sorted(dist_zipped, key=lambda x: x[1], reverse=True)[1:]
        dist_sorted, _ = zip(*dist_sorted)

        position = random.sample(dist_sorted, k=1)[0]
        return Position(row=position[0], column=position[1])

    def empty_cell_in_direction(self, direction: Direction):
        next_cell = self.agent_pos + DIR_TO_VEC[DIR_TO_INT[direction]]
        if self.within_grid(Position(row=next_cell[0], column=next_cell[1])):
            next_cell_object = self.grid.get(*next_cell)
            return not next_cell_object
        else:
            return False

    def has_object(self, object_str: str) -> bool:
        if object_str not in self._object_lookup_table.keys():
            return False
        else:
            return True

    def object_positions(self, object_str: str, object_size=None) -> List[Position]:
        assert self.has_object(object_str), "Trying to get an object's position that is not placed in the world."
        object_locations = self._object_lookup_table[object_str]
        if object_size:
            present_object_sizes = [size for size, objects in object_locations.items() if objects]
            present_object_sizes.sort()
            assert len(present_object_sizes) >= 2, "referring to a {} object but only one of its size present.".format(
                object_size)
            # Perhaps just keep track of smallest and largest object in world
            if object_size == "small":
                object_locations = object_locations[present_object_sizes[0]]
            elif object_size == "big":
                object_locations = object_locations[present_object_sizes[-1]]
            else:
                raise ValueError("Wrong size in term specifications.")
        else:
            object_locations = object_locations.items()
        return object_locations

    def step(self, action):
        self._observed_situations.append(self.get_current_situation())
        self.act(action=self.actions[action])
        new_situation = self.get_current_situation()
        reward, done = self._reward(self._observed_situations[-1], action, new_situation)
        # self._observed_situations.append(new_situation)
        return reward, done

    def render_situation(self, mission='', save_path=None, save_fig=False,
                         weight='light', actions='', countdown=''):
        if save_fig:
            # file_save_path = os.path.join(save_path, 'mission_{}, step_{:02d}.png'.format(mission, self.step_count))
            file_save_path = os.path.join(save_path, 'step_{:02d}.png'.format(self.step_count))
            image = self.render(mode="human", mission=mission, actions=actions,
                                weight=weight, countdown=countdown).getFullScreen(file_save_path=file_save_path)
        else:
            image = self.render(mode="human", mission=mission, actions=actions,
                                weight=weight, countdown=countdown).getArray()
        plt.imshow(image)

    def get_current_situation(self) -> Situation:
        return Situation(grid_size=self.grid_size,
                         agent_position=Position(row=self.agent_pos[0], column=self.agent_pos[1]),
                         target_object=self._target_object,
                         agent_direction=INT_TO_DIR[self.agent_dir],
                         placed_objects=self._placed_object_list.copy())

    def get_current_observations(self):
        return self._observed_situations.copy()

    def clear_situation(self):
        self._object_lookup_table.clear()
        self._placed_object_list.clear()
        self._occupied_positions.clear()
        self._observed_situations.clear()
        self.reachable.clear()
        self.carrying = None

    @staticmethod
    def object_displaced(current_situation: Situation, next_situation: Situation):
        shift = next_situation.target_object.position[0] - current_situation.target_object.position[0] + \
                next_situation.target_object.position[1] - current_situation.target_object.position[1]
        if abs(shift) == 1:
            return True
        else:
            return False

    @staticmethod
    def approaching_target(current_situation: Situation, next_situation: Situation):
        return -(next_situation.distance_to_target - current_situation.distance_to_target)

    def _reward(self, prev_situation, action, current_situation):
        """
        :param current_situation:
        :param action:
        :param prev_situation:
        :return: reward, done Flag
        """
        if self.step_count < self.episode_len:
            if current_situation.distance_to_target == 0:
                if self.target_verb == 'walk':
                    return 1.0, True
                elif self.target_verb in ['push', 'pull']:
                    if action == self.target_verb:
                        if self.object_displaced(prev_situation, current_situation):
                            return 1.0, True
                        else:
                            return 0.0, False
                    else:
                        return 0.0, False
                elif self.target_verb == 'pickup':
                    if action == self.target_verb:
                        return 1.0, True
                    else:
                        return 0.0, False
            else:
                return 0.0, False

        elif self.step_count == self.episode_len:
            if current_situation.distance_to_target == 0:
                if self.target_verb == 'walk':
                    return 1.0, True
                elif self.target_verb in ['push', 'pull']:
                    if action == self.target_verb:
                        if self.object_displaced(prev_situation, current_situation):
                            return 1.0, True
                        else:
                            return 0.0, True
                    else:
                        return 0.0, True
                elif self.target_verb == 'pickup':
                    if action == self.target_verb:
                        return 1.0, True
                    else:
                        return 0.0, True
            else:
                return 0.0, True
