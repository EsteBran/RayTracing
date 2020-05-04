#include <iostream>
#include "ray.h"

//sees if a specific ray hits the sphere and returns value signifying 
double hit_sphere(const point3& center, double radius, const ray& r) {
    vec3 oc = r.origin() - center;
    auto a = dot(r.direction(), r.direction());
    auto b = 2.0 * dot(oc, r.direction());
    auto c = dot(oc, oc) - radius * radius;
    auto discriminant = b * b - 4 * a * c;
    
    if (discriminant < 0) {
        return -1.0;
    }
    else {
        return (-b - sqrt(discriminant)) / (2.0 * a);
    }
}

//returns the color of each ray once it reaches the camera
color ray_color(const ray& r) {
    auto t = hit_sphere(point3(0, 0, -1), 0.5, r);
    if (t > 0.0) {
        vec3 N = unit_vector(r.at(t) - vec3(0, 0, -1));
        return 0.5 * color(N.x() + 1, N.y() + 1, N.z() + 1);
    }
    
    vec3 unit_direction = unit_vector(r.direction());
    t = 0.5 * (unit_direction.y() + 1.0);
    return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}



int main()
{
    const auto aspect_ratio = 16.0 / 9.0;
    const int image_width = 600;
    const int image_height = static_cast<int>(image_width / aspect_ratio);

    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    //defines the frustrum and the horizontal bounds
    vec3 horizontal(4.0, 0.0, 0.0);
    vec3 vertical(0.0, 2.25, 0.0);
    point3 origin(0.0, 0.0, 0.0);
    point3 lower_left_corner = origin - horizontal / 2 - vertical / 2 - vec3(0, 0, 1);

    for (int j = image_height - 1; j >= 0; j--) {
        std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
        for (int i = 0; i < image_width; i++) {
            auto u = double(i) / image_width;
            auto v = double(j) / image_height;
            ray r(origin, lower_left_corner + u * horizontal + v * vertical);
            color color = ray_color(r);
            color.write_color(std::cout);

        }
    }
    std::cerr << "\nDone.\n";


}


