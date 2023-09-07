from flask import Flask, render_template, request, redirect, jsonify
import pandas as pd
import numpy as np
import json
import random
from sklearn.cluster import KMeans

app = Flask(__name__)

DSSP = {}

config = {}
with open("config.json") as f:
    config = json.load(f)

class Node :
    counter = 1
    def __init__(self):
        self.nid = Node.counter
        self.children=[]
        self.width = 100
        Node.counter += 1
    def toDict(self):
        ret = {"nid": self.nid}
        if hasattr(self, "parent"): ret["parent"] = self.parent
        jc = []
        for c in self.children:
            jc.append(c.toDict())
        ret["children"] = jc
        if hasattr(self, "definition"): ret["definition"] = self.definition
        if hasattr(self, "description"): ret["description"] = self.description
        if hasattr(self, "name"): ret["name"] = self.name
        ret["width"] = self.width
        ret["size"] = self.size
        return ret
    def findNode(self, nid):
        if self.nid == nid: return self
        else:
            for c in self.children:
                fn = c.findNode(nid)
                if fn: return fn
        return False
    def createDF(self, pdf, root, sibling=None):
        # TODO: take or into account and numeric definition
        if not hasattr(self, "parent"): return pdf
        if hasattr(self, "definition"):
            pdf = root.findNode(self.parent).createDF(pdf,root)
            d = self.definition
            if d["type"] == "categorial" or d["type"] == "complement":
                cond = pdf[d["variable"]] == d["values"][0]
                for val in d["values"][1:]:
                    cond = cond | (pdf[d["variable"]] == val)
                return pdf[cond]
            elif d["type"] == "&gt;=":
                print(pdf[pdf[d["variable"]] >= d["value"]])
                return pdf[pdf[d["variable"]] >= d["value"]]
            elif d["type"] == "&lt;=":
                return pdf[pdf[d["variable"]] <= d["value"]]
            elif d["type"] == ">":
                return pdf[pdf[d["variable"]] > d["value"]]
            elif d["type"] == "<":
                return pdf[pdf[d["variable"]] < d["value"]]            
        if "nodes" in pdf.columns:
            return pdf[pdf.nodes.apply(lambda x: self.nid in x)]
        return pdf
    
def processDF(df, variables):
    ndf = df.copy()
    print(ndf.count())
    ndf=df.dropna()
    # ndf.dropna(inplace=True) # TODO: this should be an option
    for v in variables:
        if (variables[v]["type"] == "ID"): pass
        if (variables[v]["type"] == "numeric"):
            ndf[v] = (ndf[v] - variables[v]["mean"]) / variables[v]["std"] # choix .        
        if (variables[v]["type"] == "categorial"):
            nc = pd.get_dummies(ndf[v], prefix=v+"._.")
            # ndf.drop(v, axis=1,inplace=True)
            for k in nc:
                nc[k] = (nc[k]-nc[k].mean())/nc[k].std() # use standardScaler quicker?
            ndf = pd.concat([ndf, nc], axis=1)
    return ndf

def process(df):
    variables = {}
    df.dropna(axis=1, thresh=0.5*len(df), inplace=True)  # tresh...
    df.dropna(inplace=True)
    for k in df:
        if df[k].dtype == np.object:
            print(k,len(df[k].unique()),len(df))
            if len(df[k].unique()) < 0.01*len(df):
                variables[k] = {"type": "categorial"}
            else:
                variables[k] = {"type": "ID"}                
        else:
            variables[k] = {"type": "numeric", "mean": float(df[k].mean()), "std": float(df[k].std()), "min": float(df[k].min()), "max": float(df[k].max())}
    ndf = processDF(df, variables)    
    return ndf,variables

# TODO : should add the cut point for numeric variables
def describe(n1,n2,variables,df,odf,root):
    if hasattr(n1, "definition") and (n1.definition["type"] == "&gt;=" or n1.definition["type"] == "&lt;=" or n1.definition["type"] == ">" or n1.definition["type"] == "<"):
        n1df = n1.createDF(odf,root,sibling=n2)
        print(n1df)
        n2df = n2.createDF(odf,root,sibling=n1)
    else:
        n1df = n1.createDF(df,root,sibling=n2)
        n2df = n2.createDF(df,root,sibling=n1)
    tokeep=[]
    for k in n1df:
        if k != "nodes" and (k not in variables or (variables[k]["type"] != "ID" and variables[k]["type"] != "categorial")):
            tokeep.append(k)
    vvs = []
    for k in tokeep:
        diff = abs(n1df[k].mean()-n2df[k].mean())
        vvs.append((diff,k))
    vvs.sort(key=lambda x:x[0], reverse=True)
    n1vars = []
    n2vars = []
    donevars = []
    for vvv in vvs:
        vv = vvv[1]
        if vv in variables:
            n1vars.append([vv,odf.loc[n1df.index,vv].mean(),odf.loc[n1df.index,vv].std()])
            n2vars.append([vv,odf.loc[n2df.index,vv].mean(),odf.loc[n2df.index,vv].std()])
        else:
            v = vv[:vv.index("._.")]            
            if v not in donevars:
                donevars.append(v)
                vc1 = odf.loc[n1df.index, v].value_counts()
                n1vars.append([v,vc1.index[0],vc1[0]/n1.size])
                vc2 = odf.loc[n2df.index, v].value_counts()
                n2vars.append([v,vc2.index[0],vc2[0]/n2.size])                
    n1.description = n1vars
    n2.description = n2vars    
    
@app.route('/ds/<did>')
def showDS(did):
    return render_template("show_ds.html", ds={"did":did, "variables":DSSP[did]["variables"], "tree":DSSP[did]["tree"].toDict()})


@app.route('/varchange', methods=["POST"])
def varchange():
    data = request.get_json(force=True)
    did = data["did"]
    var = data["var"]
    ty = data["type"]
    print(f"change {var} to {ty}")
    DSSP[did]["variables"][var]["type"] = ty # change type
    # TODO: if numerical - Try to convert and calculate mean/std/min/max
    DSSP[did]["pdf"] = processDF(DSSP[did]["odf"], DSSP[did]["variables"]) # reprocess DF
    DSSP[did]["tree"].children=[]
    return {"tree":DSSP[did]["tree"].toDict(), "vars": DSSP[did]["variables"]}

@app.route('/defineNum', methods=["POST"])
def defineNum():
    data = request.get_json(force=True)
    did = data["did"]
    nid = data["nid"]
    var = data["var"]
    val = float(data["val"])
    com = data["com"]    
    n = DSSP[did]["tree"].findNode(nid)
    p = DSSP[did]["tree"].findNode(n.parent)    
    pdf = p.createDF(DSSP[did]["odf"],DSSP[did]["tree"])

    n.definition = {"type": com, "variable": var, "value": val}
    n.children=[]
    ndf = n.createDF(DSSP[did]["odf"],DSSP[did]["tree"])  # here...
    n.size = len(ndf)
    print("n.size=",n.size)
    n.width = (n.size/p.size)*100        
    if n.width < 5 : n.width = 5.0
    n2 = Node()
    n2.parent = p.nid
    p.children=[n,n2]
    n2t = ">" if com == "&lt;=" else "<"
    n2.definition = {"type": n2t, "variable": var, "value": val}
    n2df = n2.createDF(DSSP[did]["odf"], DSSP[did]["tree"], sibling=n) # here
    n2.size = len(n2df)
    print("n2.size=", n2.size)
    n2.width = (n2.size/p.size)*100
    if n2.width < 5 : n2.width = 5.0
    # realign so width add up to 100
    snw = 0
    na = 0
    for n in p.children:
        snw += n.width
        if n.width != 5.0 : na+=1
    vta = (snw-100) / na
    for n in p.children:
        if n.width != 5.0 : n.width -= vta        
    describe(n,n2,DSSP[did]["variables"],DSSP[did]["pdf"], DSSP[did]["odf"], DSSP[did]["tree"])
    return DSSP[did]["tree"].toDict()
    
@app.route('/define', methods=["POST"])
def define():
    data = request.get_json(force=True)
    did = data["did"]
    nid = data["nid"]
    var = data["var"]
    val = data["val"]
    n = DSSP[did]["tree"].findNode(nid)
    p = DSSP[did]["tree"].findNode(n.parent)    
    pdf = p.createDF(DSSP[did]["odf"],DSSP[did]["tree"])

    vals = val.split(" OR ")
    n.definition = {"type": "categorial", "variable": var, "values": vals}

    ocns = [] # get other explicitly defined concepts, and remove existing complements
    for sn in p.children[:]:
        if hasattr(sn, "definition") and "complement" not in sn.definition["type"] and sn.nid != n.nid:
            ocns.append(sn)
        elif hasattr(sn, "definition") and "complement" in sn.definition["type"]:
            p.children.remove(sn)

    svals = []
    for sn in ocns:
        for v in sn.definition["values"]:
            if v not in svals: svals.append(v)

    print(svals)
            
    others = []
    for oval in pdf[var].unique():
        if oval not in vals and oval not in svals:
            others.append(oval)

    print(others)

    # TODO: recalculate OCNS.size? (should not have changed...)
    
    n.children=[]
    ndf = n.createDF(DSSP[did]["pdf"],DSSP[did]["tree"]) 
    n.size = len(ndf)
    print("n.size=",n.size)
    n.width = (n.size/p.size)*100
    if n.width < 5 : n.width = 5.0
    n2 = Node()
    n2.parent = p.nid
    p.children=[n,n2]+ocns    
    n2.definition = {"type": "complement", "variable": var, "node": n.nid, "values": others}
    n2df = n2.createDF(DSSP[did]["pdf"], DSSP[did]["tree"], sibling=n)
    n2.size = len(n2df)
    n2.width = (n2.size/p.size)*100
    if n2.width < 5 : n2.width = 5.0

    # realign so width add up to 100
    snw = 0
    na = 0
    for n in p.children:
        snw += n.width
        if n.width != 5.0 : na+=1
    vta = (snw-100) / na
    for n in p.children:
        if n.width != 5.0 : n.width -= vta        
    describe(n,n2,DSSP[did]["variables"],DSSP[did]["pdf"], DSSP[did]["odf"], DSSP[did]["tree"])
    return DSSP[did]["tree"].toDict()

@app.route('/tFromA', methods=["POST"])
def tFromA():
    data = request.get_json(force=True)
    did = data["did"]
    nid = data["nid"]
    var = data["var"]
    avg = float(data["avg"])
    n = DSSP[did]["tree"].findNode(nid)
    df = n.createDF(DSSP[did]["odf"], DSSP[did]["tree"])    
    # find the comparator
    comp = ">="
    if avg < df[var].mean(): comp = "<="
    # find the treshold
    df=df.sort_values(var, ascending=(comp=="<="))
    index = 1
    done = False
    while not done:
        done = (comp == "<=" and df.iloc[0:index][var].mean() > avg) or (comp == ">=" and df.iloc[0:index][var].mean() < avg)
        if not done:
            index+=1
    print(df.iloc[0:index-1][var].mean())
    treshold=df.iloc[index-2][var]
    print(treshold)
    return {"treshold": int(treshold), "size": round((index-1)*100/len(df)), "comparator": comp}

@app.route('/aFromT', methods=["POST"])
def aFromT():
    data = request.get_json(force=True)
    did = data["did"]
    nid = data["nid"]
    var = data["var"]
    tre = float(data["tre"])
    com = data["com"]
    n = DSSP[did]["tree"].findNode(nid)
    df = n.createDF(DSSP[did]["odf"], DSSP[did]["tree"])    
    print(com)
    if com == "&gt;=":
        sdf = df[df[var] >= tre]
        average = sdf[var].mean()
        size=round(len(sdf)*100/len(df))
    else:
        sdf = df[df[var] <= tre]
        average = sdf[var].mean()
        size=round(len(sdf)*100/len(df))        
    return {"average": average, "size": size}

# BUG crashes when recutting root
@app.route('/cut', methods=["POST"])
def cut():
    data = request.get_json(force=True)
    did = data["did"]
    nid = data["nid"]
    print("cutting",did,nid)
    n = DSSP[did]["tree"].findNode(nid)
    variables = DSSP[did]["variables"]
    if hasattr(n, "definition") and (n.definition["type"] == "&gt;=" or n.definition["type"] == "&lt;=" or n.definition["type"] == ">" or n.definition["type"] == "<"):
        pdf = n.createDF(DSSP[did]["odf"], DSSP[did]["tree"])    
    else:
        pdf = n.createDF(DSSP[did]["pdf"], DSSP[did]["tree"])    
    tokeep = []
    for k in pdf:
        if k != "nodes" and (k not in variables or (variables[k]["type"] != "ID" and variables[k]["type"] != "categorial")):
            tokeep.append(k)
            # print(k)
    kmeans = KMeans(n_clusters=2).fit(pdf[tokeep])
    n1 = Node()
    n1.parent = n.nid
    n2 = Node()
    n2.parent = n.nid
    gpdf = DSSP[did]["pdf"]
    if "nodes" not in gpdf.columns:
        gpdf["nodes"] = [[] for i in range(len(pdf))]
    # there should be a more efficient way to do this
    labels = kmeans.labels_
    n1c = 0
    n2c = 0
    for i,ind in enumerate(pdf.index):
        if labels[i] == 0:
            n1c += 1
            gpdf.loc[ind]["nodes"].append(n1.nid)
        else:
            n2c += 1
            gpdf.loc[ind]["nodes"].append(n2.nid)
    n1.size = n1c
    n1.width = (n1c/n.size)*100
    n2.size = n2c
    n2.width = (n2c/n.size)*100
    describe(n1,n2,variables,gpdf,DSSP[did]["odf"], DSSP[did]["tree"])
    n.children = [n1,n2]
    return DSSP[did]["tree"].toDict()

@app.route('/name', methods=["POST"])
def name():
    data = request.get_json(force=True)
    did = data["did"]
    nid = data["nid"]
    name = data["name"]    
    n = DSSP[did]["tree"].findNode(nid)
    n.name = name
    return {"status": "name changed"}

@app.route('/', methods=["GET", "POST"])
def start():
    if request.method=='GET':
        return render_template('index.html')
    else:
        if 'csvfile' not in request.files:
            return render_template('index.html', error="You need to select a valid file.")
        file = request.files['csvfile']
        if not file:
            return render_template('index.html', error="You need to select a valid file.")
        if file.filename == '':
            return render_template('index.html', error="You need to select a valid file.")
        if file :
            file.save("uploads/"+file.filename)
            df = pd.read_csv("uploads/"+file.filename)
            ndf,vdf = process(df)
            did = random.getrandbits(32)
            DSSP[str(did)] = {"variables": vdf, "odf": df, "pdf": ndf, "tree": Node()}
            DSSP[str(did)]["tree"].size = len(ndf)
            return redirect(f"/ds/{did}")

if __name__ == "__main__":
    app.run(debug=True)
