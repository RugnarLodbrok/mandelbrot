from collections import Counter

import numpy as np
from kivy.app import App
from kivy.graphics.texture import Texture
from kivy.graphics.vertex_instructions import Rectangle
from kivy.uix.boxlayout import BoxLayout
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
    # print(Counter(r.reshape(64 * 64)))
    return np.moveaxis(np.stack([r, r, r]), 0, -1)


class MandelbrotScene:
    def __init__(self, center=(0, 0), r=2., rez=256, depth=64):
        self.center = np.array(center, dtype=np.float64)
        self.r = r
        self.rez = rez
        self.depth = depth

    def get_data(self):
        x, y = self.center
        r = self.r
        _, _, arr = mandelbrot_set(x - r, y - r, x + r, y + r, self.rez, self.rez, self.depth)
        return arr


class MyBoxLayout(BoxLayout):
    def __init__(self, **kwargs):
        rez = 256
        self.mandelbrot_scene = MandelbrotScene(rez=rez)
        # self.mandelbrot_scene = MandelbrotScene((-0.5, 0.5), 0.5, rez=rez)
        self.my_texture = Texture.create(size=(rez, rez), colorfmt="rgb")

        super().__init__(**kwargs)

        with self.canvas.after:
            s = 600
            self.rect = Rectangle(size=(s, s), pos=self.pos, texture=self.get_texture())

    def get_texture(self):
        print('get_texture')
        arr = self.mandelbrot_scene.get_data()

        self.my_texture.blit_buffer(to_bmp(arr).tobytes(), bufferfmt='ubyte', colorfmt='rgb')
        return self.my_texture

    def on_touch_down(self, touch):
        m = self.mandelbrot_scene
        c = (np.array(touch.pos) - np.array(self.rect.pos)) / np.array(self.rect.size) - .5
        msg = str(m.center) + "->"
        m.center += c * m.r
        print(msg, m.center)
        m.r /= 2
        self.rect.texture = self.get_texture()


class FucktrallApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self):
        return MyBoxLayout()


def on_resize(*args):
    print("resize", *args)


def main():
    Window.bind(on_resize=on_resize)
    return FucktrallApp().run()


if __name__ == '__main__':
    main()
