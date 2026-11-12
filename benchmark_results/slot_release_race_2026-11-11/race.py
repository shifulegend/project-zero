import http.client, json, subprocess, sys, time, collections, os
d, port, n, mode = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
srv = subprocess.Popen([f"{d}/pz","--model",os.path.abspath(f"{d}/../models/smollm2-q8_0.gguf"),"--server","--port",str(port),"--threads","1"],
                       stdout=open(f"/tmp/srv_{port}.log","wb"), stderr=subprocess.STDOUT, cwd=d)
try:
    for _ in range(120):
        try:
            c=http.client.HTTPConnection("127.0.0.1",port,timeout=3); c.request("GET","/health"); r=c.getresponse(); r.read(); c.close()
            if r.status==200: break
        except Exception: time.sleep(0.5)
    body=json.dumps({"messages":[{"role":"user","content":"Say hi"}],"max_tokens":2,"temperature":0,"stream":mode!="json"})
    codes=collections.Counter()
    for i in range(n):
        c=http.client.HTTPConnection("127.0.0.1",port,timeout=60)
        c.request("POST","/v1/chat/completions",body,{"Content-Type":"application/json"}); r=c.getresponse()
        if mode=="stream_done":
            while True:
                l=r.fp.readline()
                if not l or l.strip()==b"data: [DONE]": break
        else: r.read()
        c.close(); codes[r.status]+=1
    print(os.path.basename(d),mode,"N=",n,dict(codes),"429=",codes.get(429,0))
finally:
    srv.terminate()
