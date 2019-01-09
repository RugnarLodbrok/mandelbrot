from itertools import chain

import numpy as np
from kivy.app import App
from kivy.config import Config
from kivy.graphics.texture import Texture
from kivy.graphics.transformation import Matrix
from kivy.graphics.vertex_instructions import Rectangle
from kivy.uix.scatter import Scatter

Config.set('graphics', 'width', '600')
Config.set('graphics', 'height', '600')
from kivy.core.window import Window

from numba import jit


@jit
def mandelbrot(c, maxiter):
    z = c
    for n in range(maxiter):
        if abs(z) > 2:
            return n
        z = z * z + c
    return 0


@jit
def mandelbrot_set(xmin, ymin, xmax, ymax, width, height, maxiter):
    r1 = np.linspace(xmin, xmax, width)
    r2 = np.linspace(ymin, ymax, height)
    n3 = np.empty((width, height))
    for i in range(width):
        for j in range(height):
            n3[j, i] = mandelbrot(r1[i] + 1j * r2[j], maxiter)
    return (r1, r2, n3)


def to_bmp(m):
    a = np.min(m)
    b = np.max(m)
    r = ((m - a) * 255 / (b - a + 1)).astype(np.uint8)
    return np.moveaxis(np.stack([r, r, r]), 0, -1)


def transform_vector(m, v):
    mat = np.array(m.get()).reshape((4, 4))
    vec = np.array(list(chain(v, (0, 1))))
    vec2 = mat.T @ vec
    return vec2[:2]


class MandelbrotScene:
    def __init__(self, rez=256, depth=64):
        self.matrix = Matrix().translate(-1.5, -1, 0).scale(2, 2, 1)
        self.rez = rez
        self.depth = depth

    def get_data(self):
        p0 = transform_vector(self.matrix, (0, 0))
        p1 = transform_vector(self.matrix, (1, 1))
        _, _, arr = mandelbrot_set(*p0, *p1, self.rez, self.rez, self.depth)
        return arr

    def apply_transform(self, transform):
        pass


class MyScatter(Scatter):
    def __init__(self, **kwargs):
        rez = 256
        self.mandelbrot_scene = MandelbrotScene(rez=rez)
        self.my_texture = Texture.create(size=(rez, rez), colorfmt="rgb")
        super().__init__(**kwargs)
        self.do_rotation = False
        with self.canvas.before:
            s = 600
            self.rect = Rectangle(size=(s, s), pos=self.pos, texture=self.get_texture())
        self.view_matrix = Matrix().scale(s, s, 1).translate(0, 0, 0)

    def get_texture(self):
        arr = self.mandelbrot_scene.get_data()

        self.my_texture.blit_buffer(to_bmp(arr).tobytes(), bufferfmt='ubyte', colorfmt='rgb')
        return self.my_texture

    def on_touch_down(self, touch):
        if touch.is_double_tap:
            self.on_double_tap(touch)
        else:
            return super().on_touch_down(touch)

    def on_touch_up(self, touch):
        if touch.grab_current is not None:
            print('up!', self.pos, self.scale, )
        r = super().on_touch_up(touch)
        if not self._touches:
            m = self.mandelbrot_scene
            V = self.view_matrix
            T = self.transform
            # M = V @ T @ V.inv() @ M
            print(T)
            m.matrix = m.matrix.multiply(V.inverse().multiply(T.inverse()).multiply(V))
            self.rect.texture = self.get_texture()
            self.transform = Matrix()  # self.transform.identity() works strangely
        return r

    def on_double_tap(self, touch):
        # TODO: this is broken now
        pos = np.array(touch.pos)
        m = self.mandelbrot_scene
        c = (pos - np.array(self.rect.pos)) / np.array(self.rect.size) - .5
        c *= 2
        msg = str(m.matrix) + "->\n"
        m.matrix.translate(*c, 0)
        print(msg, m.matrix)
        m.matrix = m.matrix.scale(.5, .5, 1)
        self.rect.texture = self.get_texture()


class FucktrallApp(App):
    def build(self):
        return MyScatter()


def on_resize(*args):
    print("resize", *args)


def main():
    Window.bind(on_resize=on_resize)
    return FucktrallApp().run()


if __name__ == '__main__':
    main()
