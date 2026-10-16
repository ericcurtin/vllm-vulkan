// SPDX-License-Identifier: Apache-2.0
// Shared scalar decoder for embedding and projection. Buffer 0 is always raw
// uint32 storage: F16/BF16/FP8 and GGML blocks need no narrow shader features.
layout(binding=0) readonly buffer W { uint w[]; };
layout(binding=1) readonly buffer S { float scales[]; };
layout(binding=2) readonly buffer B { float biases[]; };
layout(push_constant) uniform P {
    uint width; uint rows; uint format; uint bits;
    uint group_size; uint per_row; uint row_offset; uint token;
} p;
uint byte_at(uint i) { return (w[i/4]>>((i%4)*8))&255; }
float half_at(uint i) { return unpackHalf2x16(w[i/4])[int((i%4)/2)]; }
float e4m3(uint code) {
    uint exponent=(code>>3)&15, mantissa=code&7;
    float v=exponent==0 ? float(mantissa)/512.0 :
        (exponent==15 && mantissa==7 ? 0.0 : (1.0+float(mantissa)/8.0)*exp2(float(int(exponent)-7)));
    return (code&128)!=0 ? -v : v;
}
const float e2m1[16]=float[16](0,.5,1,1.5,2,3,4,6,-0.0,-.5,-1,-1.5,-2,-3,-4,-6);
float weight_at(uint row,uint col) {
    uint i=row*p.width+col;
    if(p.format==0) return uintBitsToFloat(w[i]);
    if(p.format==1) return half_at(2*i);
    if(p.format==2) return uintBitsToFloat(((w[i/2]>>((i%2)*16))&65535)<<16);
    if(p.format==3) {
        uint bit=col*p.bits, word=row*(p.width*p.bits/32)+bit/32, shift=bit%32;
        uint code=w[word]>>shift;
        if(shift+p.bits>32) code|=w[word+1]<<(32-shift);
        code&=(1<<p.bits)-1;
        uint g=row*(p.width/p.group_size)+col/p.group_size;
        precise float value=scales[g]*float(code);
        return value+biases[g];
    }
    if(p.format==4) { // GGML Q8_0: half d, 32 signed bytes
        uint block=(i/32)*34, q=byte_at(block+2+i%32);
        return half_at(block)*float(q>=128 ? int(q)-256 : int(q));
    }
    if(p.format==5) { // GGML Q4_0: low half-block then high half-block
        uint block=(i/32)*18, j=i%32;
        uint code=(byte_at(block+2+j%16)>>((j/16)*4))&15;
        return half_at(block)*float(int(code)-8);
    }
    if(p.format==6) { // GGML Q4_K: 256 values, eight six-bit scale/min pairs
        uint block=(i/256)*144, j=i%256, sub=j/32;
        uint sc, mn;
        if(sub<4) { sc=byte_at(block+4+sub)&63; mn=byte_at(block+8+sub)&63; }
        else {
            sc=(byte_at(block+8+sub)&15)|((byte_at(block+sub)>>6)<<4);
            mn=(byte_at(block+8+sub)>>4)|((byte_at(block+4+sub)>>6)<<4);
        }
        uint code=(byte_at(block+16+(sub/2)*32+j%32)>>((sub%2)*4))&15;
        return (half_at(block)*float(sc))*float(code)-half_at(block+2)*float(mn);
    }
    if(p.format==7) return e4m3(byte_at(i))*scales[p.per_row!=0 ? row : 0];
    // NVFP4: E2M1 code and FP32-folded block/global scale.
    uint code=(byte_at(i/2)>>((i%2)*4))&15;
    return e2m1[code]*scales[row*(p.width/p.group_size)+col/p.group_size];
}
