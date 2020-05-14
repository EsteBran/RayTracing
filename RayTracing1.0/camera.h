#pragma once

#include "utility.h"

class camera {
	public:
        //defines the frustrum and the horizontal bounds
        camera() {
            horizontal = vec3(4.0, 0.0, 0.0);
            vertical = vec3(0.0, 2.25, 0.0);
            origin = point3(0.0, 0.0, 0.0);
            lower_left_corner = point3(-2.0, -1.0, -1.0);
        }

        ray get_ray(double u, double v) const {
            return ray(origin, lower_left_corner + u * horizontal + v * vertical - origin);
        }

    private:
        point3 origin;
        point3 lower_left_corner;
        vec3 horizontal;
        vec3 vertical;
};
