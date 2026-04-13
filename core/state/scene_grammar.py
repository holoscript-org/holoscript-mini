SCENE_GRAMMAR = {
    "objects": [
        {
            "id": "string",
            "type": "sphere | cube | cylinder | ring | label",
            "position": [0.0, 0.0, 0.0],
            "color": [1.0, 1.0, 1.0],
            "secondary_color": [0.8, 0.8, 0.8],
            "size": 1.0,
            "surface_style": "plain | emissive_glow | polar_caps | earth | banded | saturn_rings",
            "bands": [
                {
                    "color": [1.0, 1.0, 1.0],
                    "width": 0.1
                }
            ],
            "ring": {
                "inner_radius_factor": 1.3,
                "outer_radius_factor": 2.4,
                "color": [0.8, 0.72, 0.55],
                "opacity": 0.75
            },
            "animation": "none | orbit",
            "orbit_center": [0.0, 0.0, 0.0],
            "orbit_speed": 0.0
        }
    ]
}
