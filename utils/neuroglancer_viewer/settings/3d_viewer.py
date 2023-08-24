def setting(viewer, segments=None):
    with viewer.txn() as state:
        state.show_slices = False
        state.cross_section_background_color = "#2e2e2e"
        state.layout = "3d"

        if segments is not None:
            state.layers[0].layer.segments = set(eval(segments))
