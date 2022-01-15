import cv2

class Animation:
    """
    Animation that works both in Colab and local Jupyter.
    Based on WebM video encoding.
    """
    def __init__(
        self, name="animation", resolution=(400, 400),
        fps=20, monochrome=False):

        self.monochrome = monochrome
        self.name = name
        self.resolution = resolution
        self.writer = cv2.VideoWriter(
            name + ".mp4", cv2.VideoWriter_fourcc(*'vp90'),
            fps, resolution, isColor=not monochrome)
        assert self.writer.isOpened()
        
    def add_image(self, image):
        assert image.shape[2] == (1 if self.monochrome else 3)
        self.writer.write(cv2.resize(
            image, self.resolution, interpolation=cv2.INTER_NEAREST))

    def display(self):
        self.writer.release()
        from IPython.display import HTML
        from base64 import b64encode
        with open(self.name + ".mp4", 'rb') as f:
            video_base64 = b64encode(f.read()).decode()
        return HTML('<video controls autoplay src="data:video/webm;base64,{}">'.format(video_base64))

class AnimationMJPG(Animation):
    """
    Less efficient but safer version of `Animation`.
    Based on Motion JPEG video encoding.
    """
    def __init__(
        self, name="animation", resolution=(400, 400),
        fps=20, monochrome=False):

        self.monochrome = monochrome
        self.name = name
        self.resolution = resolution
        self.writer = cv2.VideoWriter(
            name + ".avi", cv2.VideoWriter_fourcc(*'mjpg'),
            fps, resolution, isColor=not monochrome)
        assert self.writer.isOpened()

    def display(self):
        self.writer.release()
        from IPython.display import HTML
        from base64 import b64encode
        with open(self.name + ".avi", 'rb') as f:
            video_base64 = b64encode(f.read()).decode()
        return HTML('<a href="data:video/JPEG;base64,{}" '
                    'download="animation.avi">Click to do'
                    'wnload animation</a>'.format(video_base64))
