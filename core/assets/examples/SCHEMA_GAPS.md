# Schema Gaps Found Across 10 Example Scenes

Collected from: solar_system, human_heart, atom_hydrogen, city_skyline,
molecule_water, binary_stars, clock_face, crystal_lattice,
abstract_composition, galaxy_cluster, dna_strand, mechanical_orrery

---

## Fields That Are Missing

| Field | Needed By | Description |
|-------|-----------|-------------|
| `rotation: [rx, ry, rz]` | clock, dna, molecule | Per-object Euler rotation in degrees |
| `opacity: 0.0–1.0` | galaxy, crystal, rings | Per-object transparency |
| `scale: [sx, sy, sz]` | city_skyline | Non-uniform scale (wide buildings, tall cylinders) |
| `orbit_phase: degrees` | binary_stars, orrery | Starting angle on orbit, avoids manual position offset |
| `orbit_plane: [nx, ny, nz]` | atom, galaxy | Normal vector for orbit plane (default [0,1,0] = XZ) |
| `orbit_center_ref: "object_id"` | moon, orrery | Orbit around a moving object, not a fixed point |
| `metadata: {}` | dna | Arbitrary domain data (base pair type, element, etc.) |
| `visible: bool` | any | Toggle visibility without removing object |

---

## Structural Gaps

| Gap | Exposed By | Proposed Fix |
|-----|-----------|--------------|
| Cylinder has no orientation | molecule, clock, dna | Add `rotation` field OR `from`/`to` points for auto-orient |
| No parent-child hierarchy | moon orbiting moving planet | Add `parent: "object_id"` field |
| Repetitive objects (27 identical spheres) | crystal_lattice | Add `"type": "grid"` or `"type": "scatter"` layout construct |
| Ring scale has two sources of truth | abstract_composition | Unify: `ring.outer_radius` in world units, not as factor of `size` |
| Label color is ignored in renderer | abstract_composition | Fix primitives.py to use `obj.color` for label text |
| Mesh path is CWD-relative | abstract_composition | Define an asset root; use `"assets://meshes/foo.obj"` convention |

---

## Animation Gaps

| Gap | Exposed By |
|-----|-----------|
| Only `"orbit"` type supported | clock (needs rotation), dna (needs spin) |
| Orbit always in XZ plane | clock face (needs XY), atom (needs tilted shell) |
| No counter-clockwise notation | binary_stars (negative speed is undocumented) |
| No `"rotate"` in-place animation | any spinning object |

---

## Candidate v2 Object Schema

```json
{
  "id": "unique_string",
  "label": "Display Name",
  "type": "sphere | cube | cylinder | ring | label | mesh",
  "position": [x, y, z],
  "rotation": [rx, ry, rz],
  "scale": [sx, sy, sz],
  "color": [r, g, b],
  "secondary_color": [r, g, b],
  "opacity": 1.0,
  "size": 1.0,
  "emissive": false,
  "visible": true,
  "surface_style": "plain",
  "bands": [],
  "ring": { "inner_radius": 1.3, "outer_radius": 2.4, "color": [...], "opacity": 0.75 },
  "mesh_file": "assets://meshes/foo.obj",
  "parent": null,
  "animation": {
    "type": "none | orbit | rotate | path",
    "orbit": {
      "center": [x, y, z],
      "center_ref": null,
      "radius": null,
      "speed": 1.0,
      "phase": 0.0,
      "plane_normal": [0, 1, 0]
    },
    "rotate": {
      "axis": [0, 1, 0],
      "speed": 1.0
    }
  },
  "metadata": {}
}
```
