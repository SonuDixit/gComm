import math
import random
from collections import namedtuple
import gym
from enum import IntEnum
import numpy as np
from gym import spaces
from gym.utils import seeding
from .rendering import Renderer

Position = namedtuple("Position", "row column")
# Size in pixels of a cell in the full-scale human view
CELL_PIXELS = 60

# Map of color names to RGB values
COLORS = {
    'red': np.array([255, 77, 77]),
    'green': np.array([0, 204, 102]),
    'blue': np.array([51, 153, 255]),
    'yellow': np.array([255, 235, 0]),
    'grey': np.array([100, 100, 100]),
    'pink': np.array([255, 142, 255]),
    'dimgrey': np.array([170, 170, 170])
}

# lights_out
DARK_COLORS = {
    'red': np.array([123, 0, 0]),
    'green': np.array([31, 92, 31]),
    'blue': np.array([0, 61, 115]),
    'yellow': np.array([164, 151, 38]),
    'grey': np.array([111, 111, 111]),
    'pink': np.array([206, 54, 206]),
    'dimgrey': np.array([80, 80, 80])
}

# Map of object type to integers
OBJECT_TO_IDX = {
    'unseen': 0,
    'empty': 1,
    'circle': 2,
    'cylinder': 3,
    'square': 4,
    'agent': 5,
    'diamond': 6
}

IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

# Map of agent direction indices to vectors
DIR_TO_VEC = [
    np.array((0, 1)),
    np.array((1, 0)),
    np.array((0, -1)),
    np.array((-1, 0)),
]

WEIGHT_TO_MOMENTUM = {
    "light": 1,
    "heavy": 2
}


class WorldObj:
    """
    Base class for grid world objects
    """

    def __init__(self, type, color, size=1, vector_representation=None, object_representation=None, target=False,
                 weight="light", lights_out=False):
        assert 1 <= size <= 4, "Sizes outside of range [1,4] not supported."
        self.type = type
        self.color = color
        self.border_color = color
        self.contains = None
        self.size = size

        # Initial position of the object
        self.init_pos = None

        # Current position of the object
        self.cur_pos = None

        # Representations
        self.vector_representation = vector_representation
        self.object_representation = object_representation

        # Boolean whether an object is a target
        self.target = target

        # Determining whether a heavy object can be moved in the next step or not
        self.momentum = 0
        self.weight = weight
        self.momentum_threshold = WEIGHT_TO_MOMENTUM[self.weight]

        # light or dark grid
        self.lights_out = lights_out

    def can_overlap(self):
        """Can the agent overlap with this?"""
        return True

    def can_pickup(self):
        """Can the agent pick this up?"""
        return False

    def can_push(self):
        """Can the agent push this?"""
        return False

    def render(self, r):
        """Draw this object with the given renderer"""
        raise NotImplementedError

    def _set_color(self, r):
        """Set the color of this object as the active drawing color"""
        if self.lights_out is False:
            c = COLORS[self.color]
            border_color = COLORS[self.border_color]
        else:  # lights off (hence darker grid)
            c = DARK_COLORS[self.color]
            border_color = DARK_COLORS[self.border_color]
        r.setLineColor(border_color[0], border_color[1], border_color[2])
        r.setColor(c[0], c[1], c[2])


class Square(WorldObj):
    def __init__(self, color='grey', size=1, vector_representation=None, object_representation=None, target=False,
                 weight="light", lights_out=False):
        super(Square, self).__init__('square', color, size, vector_representation=vector_representation,
                                     object_representation=object_representation, target=target, weight=weight,
                                     lights_out=lights_out)

    def render(self, r):
        self._set_color(r)

        # max_size is 4 here hardcoded
        r.drawPolygon([
            (0, CELL_PIXELS * (self.size / 4)),
            (CELL_PIXELS * (self.size / 4), CELL_PIXELS * (self.size / 4)),
            (CELL_PIXELS * (self.size / 4), 0),
            (0, 0)
        ])

    def can_pickup(self):
        """Can the agent pick this up?"""
        return True

    def can_push(self):
        return True

    def push(self):
        self.momentum += 1
        if self.momentum >= self.momentum_threshold:
            self.momentum = 0
            return True
        else:
            return False


class Cylinder(WorldObj):
    def __init__(self, color='blue', size=1, vector_representation=None, object_representation=None, target=False,
                 weight="light", lights_out=False):
        super(Cylinder, self).__init__('cylinder', color, size, vector_representation=vector_representation,
                                       object_representation=object_representation, target=target, weight=weight,
                                       lights_out=lights_out)

    def render(self, r):
        self._set_color(r)

        # Vertical quad
        parallelogram_width = (CELL_PIXELS / 2) * (self.size / 4)
        parallelogram_height = CELL_PIXELS * (self.size / 4)
        r.drawPolygon([
            (CELL_PIXELS / 2, 0),
            (CELL_PIXELS / 2 + parallelogram_width, 0),
            (CELL_PIXELS / 2, parallelogram_height),
            (CELL_PIXELS / 2 - parallelogram_width, parallelogram_height)
        ])

    def can_pickup(self):
        return True

    def can_push(self):
        return True

    def push(self):
        self.momentum += 1
        if self.momentum >= self.momentum_threshold:
            self.momentum = 0
            return True
        else:
            return False


class Circle(WorldObj):
    def __init__(self, color='blue', size=1, vector_representation=None, object_representation=None, target=False,
                 weight="light", lights_out=False):
        super(Circle, self).__init__('circle', color, size, vector_representation,
                                     object_representation=object_representation, target=target, weight=weight,
                                     lights_out=lights_out)

    def can_pickup(self):
        return True

    def can_push(self):
        return True

    def render(self, r):
        self._set_color(r)
        r.drawCircle(CELL_PIXELS * 0.5, CELL_PIXELS * 0.5, CELL_PIXELS // 10 * self.size)

    def push(self):
        self.momentum += 1
        if self.momentum >= self.momentum_threshold:
            self.momentum = 0
            return True
        else:
            return False


class Diamond(WorldObj):
    def __init__(self, color='grey', size=1, vector_representation=None, object_representation=None, target=False,
                 weight="light", lights_out=False):
        super(Diamond, self).__init__('diamond', color, size, vector_representation=vector_representation,
                                      object_representation=object_representation, target=target, weight=weight,
                                      lights_out=lights_out)

    def render(self, r):
        self._set_color(r)

        r.drawPolygon([
            (0.5 * CELL_PIXELS * (self.size / 4), 0),
            (CELL_PIXELS * (self.size / 4), 0.5 * CELL_PIXELS * (self.size / 4)),
            (0.5 * CELL_PIXELS * (self.size / 4), CELL_PIXELS * (self.size / 4)),
            (0, 0.5 * CELL_PIXELS * (self.size / 4))
        ])

    def can_pickup(self):
        return True

    def can_push(self):
        return True

    def push(self):
        self.momentum += 1
        if self.momentum >= self.momentum_threshold:
            self.momentum = 0
            return True
        else:
            return False


class Wall(WorldObj):
    def __init__(self, vector_representation=None, object_representation=None, lights_out=False):
        super().__init__('wall', color='dimgrey', vector_representation=vector_representation,
                         object_representation=object_representation, lights_out=lights_out)

    def render(self, r):
        self._set_color(r)
        r.drawPolygon([
            (0, CELL_PIXELS),
            (CELL_PIXELS, CELL_PIXELS),
            (CELL_PIXELS, 0),
            (0, 0)
        ])

        # add 3D effect
        if self.lights_out is True:
            r.setColor(100, 100, 100)
            r.setLineColor(100, 100, 100)
            r.drawPolygon([
                (0, 0),
                (CELL_PIXELS / 40, 0),
                (CELL_PIXELS / 40, CELL_PIXELS),
                (0, CELL_PIXELS)
            ])
            r.drawPolygon([
                (0, 0),
                (CELL_PIXELS, 0),
                (CELL_PIXELS, CELL_PIXELS / 40),
                (CELL_PIXELS / 40, CELL_PIXELS / 40)
            ])

            r.setColor(0, 0, 0)
            r.setLineColor(0, 0, 0)
            r.drawPolygon([
                (39.9 * CELL_PIXELS / 40, CELL_PIXELS / 40),
                (CELL_PIXELS, 0),
                (CELL_PIXELS, CELL_PIXELS),
                (39.9 * CELL_PIXELS / 40, CELL_PIXELS)
            ])
            r.drawPolygon([
                (0, CELL_PIXELS),
                (39.9 * CELL_PIXELS / 40, CELL_PIXELS),
                (39.9 * CELL_PIXELS / 40, 39.9 * CELL_PIXELS / 40),
                (CELL_PIXELS / 40, 39.9 * CELL_PIXELS / 40)
            ])

        else:  # if self.lights_out is False:
            r.setColor(220, 220, 220)
            r.setLineColor(220, 220, 220)
            r.drawPolygon([
                (0, 0),
                (CELL_PIXELS / 10, 0),
                (CELL_PIXELS / 10, CELL_PIXELS),
                (0, CELL_PIXELS)
            ])
            r.drawPolygon([
                (0, 0),
                (CELL_PIXELS, 0),
                (CELL_PIXELS, CELL_PIXELS / 10),
                (CELL_PIXELS / 10, CELL_PIXELS / 10)
            ])

            r.setColor(120, 120, 120)
            r.setLineColor(120, 120, 120)
            r.drawPolygon([
                (CELL_PIXELS - CELL_PIXELS / 10, CELL_PIXELS / 10),
                (CELL_PIXELS, 0),
                (CELL_PIXELS, CELL_PIXELS),
                (CELL_PIXELS - CELL_PIXELS / 10, CELL_PIXELS)
            ])
            r.drawPolygon([
                (0, CELL_PIXELS),
                (CELL_PIXELS - CELL_PIXELS / 10, CELL_PIXELS),
                (CELL_PIXELS - CELL_PIXELS / 10, CELL_PIXELS - CELL_PIXELS / 10),
                (CELL_PIXELS / 10, CELL_PIXELS - CELL_PIXELS / 10)
            ])

    def can_overlap(self):
        """Can the agent overlap with this?"""
        return False


class Grid:
    """
    Represent a grid and operations on it
    """

    def __init__(self, width, height, depth, lights_out):
        assert width >= 3
        assert height >= 3

        self.width = width
        self.height = height
        self._num_attributes_object = depth
        self.grid = [None] * width * height
        self.lights_out = lights_out

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

    def within_grid(self, i, j):
        if 0 <= i < self.height and 0 <= j < self.width:
            return True
        else:
            return False

    def set(self, i, j, v):
        assert self.within_grid(i, j)
        self.grid[i * self.height + j] = v

    def get(self, i, j):
        assert self.within_grid(i, j), "trying to move outside grid"
        return self.grid[i * self.height + j]

    def render(self, r, tile_size):
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """
        assert r.width == self.width * tile_size
        assert r.height == self.height * tile_size

        # Total grid size at native scale
        widthPx = self.width * CELL_PIXELS
        heightPx = self.height * CELL_PIXELS

        r.push()

        # Internally, we draw at the "large" full-grid resolution, but we
        # use the renderer to scale back to the desired size
        r.scale(tile_size / CELL_PIXELS, tile_size / CELL_PIXELS)

        # Draw the background of the in-world cells black
        if self.lights_out is True:
            r.fillRect(
                0,
                0,
                heightPx,
                widthPx,
                115, 115, 115
            )
        else:
            r.fillRect(
                0,
                0,
                heightPx,
                widthPx,
                255, 255, 255
            )

        # Draw grid lines
        r.setLineColor(100, 100, 100)
        for rowIdx in range(0, self.height):
            x = CELL_PIXELS * rowIdx
            r.drawLine(x, 0, x, heightPx)
        for colIdx in range(0, self.width):
            y = CELL_PIXELS * colIdx
            r.drawLine(0, y, widthPx, y)

        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):
                cell = self.get(i, j)
                if cell is None:
                    continue
                r.push()
                r.translate(j * CELL_PIXELS, i * CELL_PIXELS)
                cell.render(r)
                r.pop()

        r.pop()


class MiniGridEnv(gym.Env):
    """
    2D grid world game environment.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array', 'pixmap'],
        'video.frames_per_second': 10
    }

    # Enumeration of possible actions
    class Actions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2
        backward = 3

        # push, pull objects
        push = 4
        pull = 5
        pickup = 6
        drop = 7

    def __init__(self, grid_size=None, width=None, height=None, max_steps=100, seed=1337):
        # Can't set both grid_size and width/height
        if grid_size:
            assert width == None and height == None
            width = grid_size
            height = grid_size

        # Action enumeration for this environment
        self.actions = MiniGridEnv.Actions

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))

        # Renderer object used to render the whole grid (full-scale)
        self.grid_render = None

        # Renderer used to render observations (small-scale agent view)
        self.obs_render = None

        # Environment configuration
        self.width = width
        self.height = height
        self.max_steps = max_steps

        # Current position and direction of the agent
        self.agent_pos = None
        self.agent_dir = None

        # Initialize the RNG
        self.seed(seed=seed)

        # Initialize the state
        self.reset()

    def reset(self):
        # Current position and direction of the agent
        self.agent_pos = None
        self.agent_dir = None

        # Item picked up, being carried, initially nothing
        self.carrying = None
        self.step_count = 0  # Step count since episode start
        return

    def seed(self, seed=1337):
        # Seed the random number generator
        self.np_random, _ = seeding.np_random(seed)
        return [seed]

    @property
    def steps_remaining(self):
        return self.max_steps - self.step_count

    def __str__(self):
        """
        Produce a pretty string of the environment's grid along with the agent.
        A grid cell is represented by 2-character string, the first one for
        the object and the second one for the color.
        """

        # Map of object types to short string
        OBJECT_TO_STR = {
            'circle': 'A',
            'square': 'B',
            'cylinder': 'C',
            'diamond': 'D'
        }

        # Map agent's direction to short string
        AGENT_DIR_TO_STR = {
            0: '>',
            1: 'V',
            2: '<',
            3: '^'
        }

        str = ''
        for j in range(self.height):
            for i in range(self.width):
                if i == self.agent_pos[0] and j == self.agent_pos[1]:
                    str += 2 * AGENT_DIR_TO_STR[self.agent_dir]
                    continue
                c = self.grid.get(i, j)
                if not c:
                    str += '  '
                    continue
                str += OBJECT_TO_STR[c.type] + c.color[0].upper()
            if j < self.height - 1:
                str += '\n'
        return str

    def place_obj(self, obj, top=None, size=None, reject_fn=None, max_tries=math.inf):
        """
        Place an object at an empty position in the grid

        :param obj:
        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
        :param max_tries:
        """

        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.width, self.height)

        num_tries = 0

        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop
            if num_tries > max_tries:
                raise RecursionError('rejection sampling failed in place_obj')

            num_tries += 1

            pos = np.array((
                np.random.randint(top[0], min(top[0] + size[0], self.width)),
                np.random.randint(top[1], min(top[1] + size[1], self.height))
            ))

            # Don't place the object on top of another object
            if self.grid.get(*pos) is not None:
                continue

            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos):
                continue

            break

        self.grid.set(*pos, obj)

        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos

    def place_agent(self, top=None, size=None, rand_dir=True, max_tries=math.inf):
        """
        Set the agent's starting point at an empty position in the grid
        """
        pos = self.place_obj(None, top, size, max_tries=max_tries)
        self.agent_pos = pos

        if rand_dir:
            self.agent_dir = np.random.randint(0, 4)

        return pos

    @property
    def dir_vec(self):
        """
        Get the direction vector for the agent, pointing in the direction
        of forward movement.
        """
        assert 0 <= self.agent_dir < 4
        return DIR_TO_VEC[self.agent_dir]

    @property
    def front_pos(self):
        """
        Get the position of the cell that is right in front of the agent
        """
        return self.agent_pos + self.dir_vec

    def act(self, action):
        self.step_count += 1

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            fwd_pos = self.front_pos  # Get the position in front of the agent
            try:
                fwd_cell = self.grid.get(*fwd_pos)  # Get the contents of the cell in front of the agent
                if fwd_cell is None or fwd_cell.can_overlap():
                    self.agent_pos = fwd_pos
            except AssertionError:  # trying to access cell outside grid
                pass

        elif action == self.actions.backward:
            bwd_pos = self.agent_pos - self.dir_vec
            try:
                bwd_cell = self.grid.get(*bwd_pos)
                if bwd_cell is None or bwd_cell.can_overlap():
                    self.agent_pos = bwd_pos
            except AssertionError:  # trying to access cell outside grid
                pass

        # Push object
        elif action == self.actions.push:
            curr_pos = self.agent_pos
            curr_cell = self.grid.get(*curr_pos)  # Get the contents of the cell of the agent
            if curr_cell is not None:
                if curr_cell.can_push():  # see if object can be pushed
                    if curr_cell.push():  # if momentum >= weight
                        fwd_pos = curr_pos + self.dir_vec  # new agent/object position

                        try:
                            fwd_cell = self.grid.get(*fwd_pos)  # contents of the position in front of the agent
                            if fwd_cell is None:  # If the new position isn't occupied by another object, push it forward
                                # self.grid.set(*fwd_pos, curr_cell)  # move object one step forward
                                self.agent_pos = fwd_pos  # agent moved one step forward
                                curr_pos = Position(row=curr_pos[0], column=curr_pos[1])
                                fwd_pos = Position(row=fwd_pos[0], column=fwd_pos[1])
                                if curr_pos == self._target_object.position:
                                    self.move_object(curr_pos, fwd_pos, target=True)
                                else:
                                    self.move_object(curr_pos, fwd_pos, target=False)

                        except AssertionError:
                            pass

        # Pull object
        elif action == self.actions.pull:
            curr_pos = self.agent_pos
            curr_cell = self.grid.get(*curr_pos)  # Get the contents of the cell of the agent
            if curr_cell is not None:
                if curr_cell.can_push():  # see if object can be pushed
                    if curr_cell.push():  # if momentum >= weight
                        bwd_pos = self.agent_pos - self.dir_vec  # new agent/object position

                        try:
                            bwd_cell = self.grid.get(*bwd_pos)
                            if bwd_cell is None:
                                # self.grid.set(*bwd_pos, curr_cell)  # move the object one step behind
                                self.agent_pos = bwd_pos  # move agent to new position
                                curr_pos = Position(row=curr_pos[0], column=curr_pos[1])
                                bwd_pos = Position(row=bwd_pos[0], column=bwd_pos[1])
                                if curr_pos == self._target_object.position:
                                    self.move_object(curr_pos, bwd_pos, target=True)
                                else:
                                    self.move_object(curr_pos, bwd_pos, target=False)
                        except AssertionError:
                            pass

        # Pickup object
        elif action == self.actions.pickup:
            curr_pos = self.agent_pos
            curr_cell = self.grid.get(*curr_pos)  # Get the contents of the cell of the agent
            if curr_cell is not None:
                if curr_cell.can_pickup():  # see if object can be picked up
                    if not self.carrying:  # if listener is not carrying any object
                        # self.carrying = curr_cell
                        curr_pos = Position(row=curr_pos[0], column=curr_pos[1])
                        self.carrying = self._remove_object(from_position=curr_pos)

        # Drop object
        elif action == self.actions.drop:
            curr_pos = self.agent_pos
            curr_cell = self.grid.get(*curr_pos)
            if curr_cell is None:
                if self.carrying:
                    curr_pos = Position(row=curr_pos[0], column=curr_pos[1])
                    if curr_pos == self._target_object.position:
                        self.place_object(self.carrying.object, curr_pos, target=True)
                    else:
                        self.place_object(self.carrying.object, curr_pos, target=False)
                    self.carrying = None

        else:
            print(action)
            assert False, "unknown action"

    def generate_maze(self, complexity=.10, density=.50):
        """
        Generate a random maze array.

        It only contains two kind of objects, obstacle and free space. The numerical value for obstacle
        is ``1`` and for free space is ``0``.

        Code from https://en.wikipedia.org/wiki/Maze_generation_algorithm
        """
        # Only odd shapes
        shape = ((self.height // 2) * 2, (self.width // 2) * 2)

        # Adjust complexity and density relative to maze-size
        complexity = int(complexity * (5 * (shape[0] + shape[1])))
        density = int(density * ((shape[0] // 2) * (shape[1] // 2)))

        # Build actual maze
        Z = np.zeros(shape, dtype=bool)

        # Make aisles
        for i in range(density):
            x, y = random.randint(0, (shape[1] - 1) // 2) * 2, random.randint(0, (shape[0] - 1) // 2) * 2
            Z[y, x] = 1
            for j in range(complexity):
                neighbours = []
                if x > 1:             neighbours.append((y, x - 2))
                if x < shape[1] - 2:  neighbours.append((y, x + 2))
                if y > 1:             neighbours.append((y - 2, x))
                if y < shape[0] - 2:  neighbours.append((y + 2, x))
                if len(neighbours):
                    y_, x_ = neighbours[random.randint(0, len(neighbours) - 1)]
                    if Z[y_, x_] == 0:
                        Z[y_, x_] = 1
                        Z[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                        x, y = x_, y_

        # print('maze: \n', Z.astype(int))
        return Z.astype(int)

    def render(self, mission='', mode='', close=False, weight='light', highlight=True, tile_size=CELL_PIXELS,
               actions='', countdown='', image_input=False, lights_out=False):
        """
        Render the whole-grid human view
        """

        if close:
            if self.grid_render:
                self.grid_render.close()
            return

        if self.grid_render is None or self.grid_render.window is None or (
                self.grid_render.width != self.width * tile_size):
            self.grid_render = Renderer(
                self.width * tile_size,
                self.height * tile_size,
                True if mode == 'human' else False
            )

        r = self.grid_render

        if image_input is not True:
            if r.window:
                if r.window:
                    if weight == 'heavy':
                        r.window.setText('Instruction: ' + mission + ' twice')
                    else:
                        r.window.setText('Instruction: ' + mission)
                r.window.setCountdown(text=str(countdown))
                r.window.setActions('Actions: ' + actions)

        r.beginFrame()

        # Render the whole grid
        self.grid.render(r, tile_size)

        # Draw the agent
        ratio = tile_size / CELL_PIXELS
        r.push()
        r.scale(ratio, ratio)
        r.translate(
            CELL_PIXELS * (self.agent_pos[1] + 0.5),
            CELL_PIXELS * (self.agent_pos[0] + 0.5)
        )
        r.rotate(self.agent_dir * 90)

        if lights_out is True:
            r.setLineColor(DARK_COLORS["pink"][0], DARK_COLORS["pink"][1], DARK_COLORS["pink"][2])
            r.setColor(DARK_COLORS["pink"][0], DARK_COLORS["pink"][1], DARK_COLORS["pink"][2])
        else:
            r.setLineColor(COLORS["pink"][0], COLORS["pink"][1], COLORS["pink"][2])
            r.setColor(COLORS["pink"][0], COLORS["pink"][1], COLORS["pink"][2])

        r.drawPolygon([
            (-12, 10),
            (12, 0),
            (-12, -10)
        ])
        r.pop()
        r.endFrame()

        if mode == 'rgb_array':
            return r.getArray()
        elif mode == 'pixmap':
            return r.getPixmap()
        return r
