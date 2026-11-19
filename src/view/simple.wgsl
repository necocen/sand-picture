// The Simple view: each pixel takes its cell's colour. Preserve the original
// cell palette and optical-depth arithmetic exactly: the behaviour is read in
// this view, and the regression's baseline holds its pixels.
@vertex fn vertex(@builtin(vertex_index) i:u32)->@builtin(position) vec4<f32> {
    let p=vec2<f32>(f32((i<<1u)&2u),f32(i&2u));return vec4<f32>(p*2.-1.,0.,1.);
}
@fragment fn fragment(@builtin(position) position:vec4<f32>)->@location(0) vec4<f32> {
    let pixel=position.xy*vec2<f32>(cfg.size)/vec2<f32>(cfg.image_size);
    let x=min(u32(pixel.x),cfg.size.x-1u);let y=cfg.size.y-1u-min(u32(pixel.y),cfg.size.y-1u);
    let c=composition(vec2<i32>(i32(x),i32(y)));
    let gas=clamp(c.w,0.,1.);let liquid=1.-gas;
    let depths=3.*.001*max(c.yz,vec2<f32>(0.))/cfg.diameters;
    let depth=depths.x+depths.y;
    let water=vec3<f32>(.00303527,.00439144,.00913406);
    var color=water;
    if depth>0. && liquid>0. {
        let sand=mix(vec3<f32>(.90466117,.77582222,.51491767),vec3<f32>(.11697067,.06124605,.02315337),depths.y/depth);
        color=mix(water,sand,1.-exp(-depth/liquid));
    }
    return vec4<f32>(liquid*color+gas*vec3<f32>(.67244316,.83879901,.92158186),1.);
}
