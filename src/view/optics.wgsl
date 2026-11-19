// The optics of Photoreal's surfaces: the refractive indices of the water
// and the air, and the Fresnel reflectance of an interface between two media.
const WATER_IOR: f32 = 1.333;
const AIR_IOR: f32 = 1.00029;

// Unpolarized dielectric Fresnel, including water-to-air total reflection.
// eta is incident / transmitted IOR (the convention used by WGSL refract).
fn fresnel(cosine: f32, eta: f32) -> f32 {
    let sin2_t = eta * eta * max(0., 1. - cosine * cosine);
    if sin2_t >= 1. { return 1.; }
    let cos_t = sqrt(1. - sin2_t);
    let rs = (eta * cosine - cos_t) / (eta * cosine + cos_t);
    let rp = (cosine - eta * cos_t) / (cosine + eta * cos_t);
    return .5 * (rs * rs + rp * rp);
}
