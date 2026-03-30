# iPZE
# creadted by pze

## Explanation of `delta` in Gaussian Heatmap Radius Calculation

In the `put_heatmap` function, the variable `delta` is the **normalized radius factor** that determines how far from the center point the Gaussian influence extends. It is derived directly from the truncation threshold `th` and is used to compute the bounding box of pixels that need to be evaluated.

### The Gaussian formula

The heatmap value at each pixel `(x, y)` is computed using a 2-D isotropic Gaussian:

```
G(x, y) = exp( -d / (2 * sigma^2) )
```

where `d = (x - center_x)^2 + (y - center_y)^2` is the squared Euclidean distance from the center.

### The truncation threshold `th`

Computing the Gaussian for every pixel in the image would be expensive. Instead, pixels whose Gaussian value falls below a minimum useful level are skipped. The threshold is set so that:

```
exp(-th) ≈ exp(-4.6052) ≈ 0.01
```

Any pixel whose exponent `d / (2 * sigma^2)` exceeds `th` produces a Gaussian value below ~1 %, so it is discarded with `continue`.

### Deriving `delta`

The cut-off condition is:

```
d / (2 * sigma^2) <= th
(x - cx)^2 + (y - cy)^2 <= th * 2 * sigma^2
```

The largest distance along a single axis from the center (i.e. the half-width of the bounding box) is therefore:

```
|axis_offset| <= sqrt(th * 2) * sigma
```

Defining:

```python
delta = math.sqrt(th * 2)   # ≈ 3.035
```

separates the pure mathematical factor `sqrt(2 * th)` from the scale factor `sigma`, making the bounding box computation explicit:

```python
x0 = int(max(0,      center_x - delta * sigma))
x1 = int(min(width,  center_x + delta * sigma))
y0 = int(max(0,      center_y - delta * sigma))
y1 = int(min(height, center_y + delta * sigma))
```

### Summary

| Symbol | Value | Meaning |
|--------|-------|---------|
| `th` | 4.6052 | Exponent threshold; `exp(-th) ≈ 0.01` |
| `delta` | `sqrt(2 * th) ≈ 3.035` | Dimensionless half-width of the influence region in units of `sigma` |
| `delta * sigma` | ~3 standard deviations | Pixel radius of the local bounding box |

`delta` therefore encapsulates the relationship between the threshold and the Gaussian spread: any pixel more than `delta * sigma` pixels away from the center is guaranteed to have a Gaussian response below 1 % and can safely be ignored.
