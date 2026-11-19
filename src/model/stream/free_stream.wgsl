// Frozen occupied-face velocity map, used by both capillary terms.
fn outside_boundary(p:vec2<i32>)->bool{
    return p.x==0||p.y==0||p.x+1==i32(cfg.nx)||p.y+1==i32(cfg.ny);
}
fn open_segment(p:vec2<i32>,a:u32)->f32{
    // a=0: horizontal psi edge -> negative vertical volume current.
    // a=1: vertical psi edge -> positive horizontal volume current.
    if a==0u{
        if p.y<=0||p.y>=i32(cfg.fy)||p.x<0||p.x>=i32(cfg.fx){return 0.;}
        let f=2u*(u32(p.y-1)*cfg.fx+u32(p.x))+1u;
        return geometry[38u*cfg.fx*cfg.fy+f].x;
    }
    if p.x<=0||p.x>=i32(cfg.fx)||p.y<0||p.y>=i32(cfg.fy){return 0.;}
    let f=2u*(u32(p.y)*cfg.fx+u32(p.x-1));
    return geometry[38u*cfg.fx*cfg.fy+f].x;
}
fn body_boundary(p:vec2<i32>)->bool{
    return open_segment(p,0u)<=0.||open_segment(p-vec2<i32>(1,0),0u)<=0.
        ||open_segment(p,1u)<=0.||open_segment(p-vec2<i32>(0,1),1u)<=0.;
}
fn segment_basis(p:vec2<i32>,q:vec2<i32>,a:u32)->f32{
    let area=open_segment(p,a);if area<=0.{return 0.;}
    let step=select(vec2<i32>(1,0),vec2<i32>(0,1),a==1u);
    let coefficient=select(0.,1.,all(q==p+step))-select(0.,1.,all(q==p));
    return select(-coefficient,coefficient,a==1u)/area;
}
fn gradient_basis(p:vec2<i32>,d:vec2<i32>)->vec4<f32>{
    let q=p+d;let x=vec2<i32>(1,0);let y=vec2<i32>(0,1);
    let xx=.25*(segment_basis(p+x,q,1u)+segment_basis(p+x-y,q,1u)
        -segment_basis(p-x,q,1u)-segment_basis(p-x-y,q,1u));
    let xy=segment_basis(p,q,1u)-segment_basis(p-y,q,1u);
    let yx=segment_basis(p,q,0u)-segment_basis(p-x,q,0u);
    let yy=.25*(segment_basis(p+y,q,0u)+segment_basis(p+y-x,q,0u)
        -segment_basis(p-y,q,0u)-segment_basis(p-y-x,q,0u));
    return vec4<f32>(xx,xy,yx,yy);
}
fn gradient_index(p:vec2<i32>,d:vec2<i32>)->u32{
    return cfg.nx*cfg.ny+9u*at(p,cfg.nx)+u32((d.y+1)*3+d.x+1);
}
fn gradient(p:vec2<i32>,d:vec2<i32>)->vec4<f32>{return tensor[gradient_index(p,d)];}
fn gram(a:vec4<f32>,b:vec4<f32>,s:vec3<f32>)->f32{
    return s.x*(a.x*b.x+a.z*b.z)
        +s.y*(a.x*b.y+a.y*b.x+a.z*b.w+a.w*b.z)
        +s.z*(a.y*b.y+a.w*b.w);
}
