"""Tiny demo page for the Lichess endpoint. Holds the key server-side.
Run on the VM:  ENDPOINT_KEY=... python3 demo_app.py serving/sample-request.json
Then open http://<VM-IP>:8080 . Stop it (Ctrl+C) and close the NSG rule after the demo.
"""
import json, os, sys, urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer

URL = os.environ.get("ENDPOINT_URL", "https://eplichess60306369.qatarcentral.inference.ml.azure.com/score")
KEY = os.environ["ENDPOINT_KEY"]
PORT = int(os.environ.get("PORT", "8080"))
SAMPLE = open(sys.argv[1]).read() if len(sys.argv) > 1 else '{"data": []}'

PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Lichess outcome predictor</title>
<style>body{font-family:Georgia,serif;background:#F3EFE3;color:#1F1D1A;max-width:900px;margin:30px auto;padding:0 16px}
h1{font-weight:400}textarea{width:100%;height:260px;font-family:monospace;font-size:13px}
button{border:0;padding:11px 22px;font-size:17px;cursor:pointer;margin:8px 8px 8px 0;font-family:inherit}
.go{background:#44602B;color:#fff}.alt{background:#fff;border:2px solid #1F1D1A;color:#1F1D1A}.bad{background:#fff;border:2px solid #A33;color:#A33}
.g{margin:22px 0 0}.g h3{margin:0 0 6px;font-weight:400}.row{display:flex;align-items:center;gap:10px;margin:4px 0}
.bar{height:24px;background:#6E8F4E}.err{background:#fbeaea;border:2px solid #A33;padding:12px;margin-top:14px}
pre{background:#fff;padding:12px;overflow:auto;font-size:12px}small{color:#6A665B}</style>
<h1>Lichess outcome predictor</h1>
<p>Paste games (16 computed features each) and press Predict. Calls the Azure ML endpoint <b>eplichess60306369</b>.</p>
<textarea id=inp>__SAMPLE__</textarea><br>
<button class=go onclick=go()>Predict</button>
<button class=alt onclick=newGame()>New test input</button>
<button class=alt onclick=reset()>Reset to sample</button>
<button class=bad onclick=bad()>Incorrect input example</button>
<div id=viz></div><details><summary><small>Raw response</small></summary><pre id=out></pre></details>
<script>
const SAMPLE=document.getElementById('inp').value;
const inp=()=>document.getElementById('inp');
const R=(a,b)=>Math.floor(a+Math.random()*(b-a+1));
const OPEN=[['D02','D',"Queen's Pawn Game"],['B01','B','Scandinavian Defense'],['C50','C','Italian Game'],['B20','B','Sicilian Defense'],['C00','C','French Defense'],['A00','A','Uncommon Opening']];
const TC=[[60,0],[180,0],[300,0],[300,3],[600,0],[600,5],[900,10],[1800,0]];
function newGame(){
 const we=R(1000,2500),be=Math.max(800,we+R(-250,250)),tc=TC[R(0,TC.length-1)],o=OPEN[R(0,OPEN.length-1)];
 const est=tc[0]+40*tc[1],sc=est<180?0:est<480?1:est<1500?2:3;
 const wu=R(30,Math.min(150,tc[0])),bu=R(30,Math.min(150,tc[0]));
 const g={id:'new-'+R(100,999),white_elo:we,black_elo:be,elo_diff:we-be,elo_mean:(we+be)/2,tc_base:tc[0],tc_inc:tc[1],speed_class:sc,eco:o[0],eco_family:o[1],opening_family:o[2],w_clk_1:tc[0]-1,b_clk_1:tc[0]-1,w_used_10:wu,b_used_10:bu,used_diff_10:wu-bu,unequal_start:0};
 inp().value=JSON.stringify({data:[g]},null,1);go();
}
function reset(){inp().value=SAMPLE;go();}
function bad(){
 let j;try{j=JSON.parse(inp().value)}catch(e){j=JSON.parse(SAMPLE)}
 const d=j.data&&j.data.length?j.data:JSON.parse(SAMPLE).data;
 const g=Object.assign({},d[0]);delete g.white_elo;
 inp().value=JSON.stringify({data:[g]},null,1);go();
}
async function go(){
 document.getElementById('out').textContent='...';document.getElementById('viz').innerHTML='';
 const r=await fetch('/predict',{method:'POST',body:inp().value});
 const t=await r.text();let j;try{j=JSON.parse(t);if(typeof j==='string')j=JSON.parse(j)}catch(e){}
 document.getElementById('out').textContent=j?JSON.stringify(j,null,2):t;
 const v=document.getElementById('viz');
 if(j&&j.probabilities){
  v.innerHTML=j.probabilities.map((p,i)=>'<div class=g><h3>'+((j.ids&&j.ids[i])||('game '+(i+1)))+' &rarr; <b>'+((j.labels&&j.labels[i])||'').replace('_',' ')+'</b></h3>'+
   ['White win','Draw','Black win'].map((n,k)=>'<div class=row><span style="width:100px">'+n+'</span><div class=bar style="width:'+Math.round(p[k]*500)+'px"></div> '+(p[k]*100).toFixed(1)+'%</div>').join('')+'</div>').join('')+
   '<p><small>Model: '+(j.model||'?')+(j.warnings&&j.warnings.length?' · warnings: '+j.warnings.join('; '):'')+'</small></p>';
 }else{
  const code=j&&(j.code||(j.error&&j.error.code)||j.error);
  v.innerHTML='<div class=err><b>Input rejected</b><br>'+(code?('error code: <b>'+(typeof code==='string'?code:JSON.stringify(code))+'</b><br>'):'')+((j&&(j.message||j.detail))||'')+'<br><small>Azure shows this as HTTP 424; the endpoint\'s own code is the one above.</small></div>';
 }
}
</script>"""

class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="text/html"):
        b = body.encode()
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        self._send(200, PAGE.replace("__SAMPLE__", SAMPLE.replace("<", "&lt;")))
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); body = self.rfile.read(n)
        req = urllib.request.Request(URL, data=body, method="POST", headers={
            "Authorization": "Bearer " + KEY, "Content-Type": "application/json",
            "azureml-model-deployment": "blue"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                self._send(200, r.read().decode(), "application/json")
        except urllib.error.HTTPError as e:
            self._send(200, e.read().decode(), "application/json")
        except Exception as e:
            self._send(200, json.dumps({"error": str(e)}), "application/json")

HTTPServer(("0.0.0.0", PORT), H).serve_forever()
