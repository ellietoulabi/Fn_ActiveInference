try:
    from IPython.display import Image, display
    from ipywidgets import IntSlider, interactive
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False
    # Dummy implementations
    def display(*args, **kwargs):
        pass
    def Image(*args, **kwargs):
        pass
    def IntSlider(*args, **kwargs):
        pass
    def interactive(*args, **kwargs):
        pass


def show_image_in_ipython(data, *args, **kwargs):
    if HAS_IPYTHON:
        display(Image(data, *args, **kwargs))
    else:
        pass  # Silently ignore if IPython not available


def ipython_images_slider(image_pathes_list, slider_label="", first_arg=0):
    if not HAS_IPYTHON:
        return None  # Return None if IPython not available
    def display_f(**kwargs):
        display(Image(image_pathes_list[kwargs[slider_label]]))

    return interactive(
        display_f,
        **{slider_label: IntSlider(min=0, max=len(image_pathes_list) - 1, step=1)},
    )


def show_ipython_images_slider(image_pathes_list, slider_label="", first_arg=0):
    if not HAS_IPYTHON:
        return  # Silently ignore if IPython not available
    def display_f(**kwargs):
        display(Image(image_pathes_list[kwargs[slider_label]]))

    display(
        interactive(
            display_f,
            **{slider_label: IntSlider(min=0, max=len(image_pathes_list) - 1, step=1)},
        )
    )
