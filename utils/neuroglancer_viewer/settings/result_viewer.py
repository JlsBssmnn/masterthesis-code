def setting(viewer, segments=None):
    with viewer.txn() as state:
        state.cross_section_background_color = "#2e2e2e"
        state.layout = "yz"
        state.crossSectionScale = 0.1

        for layer in state.layers:
            layer.visible = False
