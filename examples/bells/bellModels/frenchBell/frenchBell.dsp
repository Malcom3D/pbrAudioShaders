// 1424 broadway street
import("stdfaust.lib");
import("modalModel.lib");


exPos = nentry("exPos",0,0,6,1) : ba.sAndH(gate);
exSpread = hslider("exSpread",0,0,1,0.01) : ba.sAndH(gate);
t60Scaler = hslider("t60",1,0,100,0.01) : ba.sAndH(gate);
t60Decay = hslider("t60Decay",0.75,0,1,0.01) : ba.sAndH(gate);
t60Slope = hslider("t60Slope",2,1,6,0.01) : ba.sAndH(gate);
hammerHardness = hslider("hammerHardness",0.95,0,1,0.01) : ba.sAndH(gate);
hammerSize = hslider("hammerSize",0.3,0,1,0.01) : ba.sAndH(gate);
gain = hslider("gain",0.3,0,1,0.01);
gate = button("gate");

hammer(trig,hardness,size) = en.ar(att,att,trig)*no.noise : fi.lowpass(3,ctoff)
with{
  ctoff = (1-size)*9500+500;
  att = (1-hardness)*0.01+0.001;
};

process = hammer(gate,hammerHardness,hammerSize) : modalModel(exPos,30,1,3)*gain <: _,_;
