def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * ( 1 -factor ))
        else:
            smoothed_points.append(points)
    return smoothed_points