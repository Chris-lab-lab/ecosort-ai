"""Procedural single/multiple-object waste scenes with exact YOLO labels.

This is a synthetic detector TEST dataset, not real waste photography. Each
object silhouette is rendered with its own instance ID, so boxes can be derived
from the final visible pixels after scaling, rotation, clipping, and occlusion.
No camera photos, remote assets, or third-party product images are used.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


CLASSES = ("plastic_bottle", "metal_can", "wrapper")
PALETTE = (
    (29, 122, 180), (185, 48, 55), (34, 134, 82), (213, 156, 36),
    (126, 73, 154), (232, 105, 37), (216, 218, 204), (34, 49, 62),
)
SOURCE_DIR = Path(__file__).resolve().parent
PROJECT = SOURCE_DIR.parent


def font(size: int, bold: bool = False):
    names = ("arialbd.ttf", "DejaVuSans-Bold.ttf") if bold else ("arial.ttf", "DejaVuSans.ttf")
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def choose(rng, values):
    return values[int(rng.integers(len(values)))]


def tint(color, amount):
    return tuple(int(np.clip(component * amount, 0, 255)) for component in color)


def masked_layer(size, mask, base, rng, metallic=False, alpha=255):
    """Cylinder-like diffuse and specular shading, clipped to a silhouette."""
    width, height = size
    y, x = np.mgrid[0:height, 0:width]
    center = width * rng.uniform(0.45, 0.55)
    normal_x = np.clip((x - center) / (width * 0.46), -1, 1)
    curved = np.sqrt(np.maximum(0, 1 - normal_x**2))
    diffuse = 0.36 + 0.64 * curved
    diffuse *= 0.95 + 0.06 * np.sin(y / height * math.pi)
    specular = np.exp(-((normal_x + rng.uniform(0.25, 0.5)) / 0.11) ** 2)
    reflection = np.exp(-((normal_x - 0.55) / 0.08) ** 2) * 0.22
    noise = rng.normal(0, 1.4, (height, width, 1))
    rgb = np.array(base)[None, None, :] * diffuse[..., None]
    rgb = rgb + (specular + reflection)[..., None] * (95 if metallic else 62) + noise
    result = np.zeros((height, width, 4), dtype=np.uint8)
    result[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    result[..., 3] = (np.asarray(mask, dtype=np.float32) * alpha / 255).astype(np.uint8)
    return Image.fromarray(result)


def add_band(image, mask, box, rng, text_pool, color):
    width, height = image.size
    layer = Image.new("RGBA", image.size)
    draw = ImageDraw.Draw(layer)
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill=(*color, 255))
    # Shared graphic vocabulary across all classes prevents a fixed color cue.
    stripe = choose(rng, PALETTE)
    draw.polygon([(x0,y0),(x0+(x1-x0)*.25,y0),(x1,y1),(x1-(x1-x0)*.3,y1)], fill=(*stripe,255))
    text = choose(rng, text_pool)
    label_font = font(max(14, int((y1-y0)*.21)), True)
    ink = (243,244,234,255) if sum(color) < 430 else (32,43,48,255)
    draw.text(((x0+x1)/2,(y0+y1)/2), text, fill=ink, font=label_font, anchor="mm")
    draw.line([(x0+12,y1-10),(x1-12,y1-10)], fill=(235,235,220,190), width=2)
    band = np.array(layer)
    band[...,3] = np.minimum(band[...,3], np.asarray(mask))
    # Wrap the label illumination around the cylinder.
    light = 0.65 + .35 * np.sin(np.linspace(0, math.pi, width))
    band[...,:3] = (band[...,:3].astype(float) * light[None,:,None]).astype(np.uint8)
    image.alpha_composite(Image.fromarray(band))


def bottle(rng):
    width, height = 220, 400
    mask = Image.new("L", (width,height))
    draw_mask = ImageDraw.Draw(mask)
    neck = int(rng.integers(37,57))
    shoulder = int(rng.integers(78,112))
    left, right = int(rng.integers(30,44)), int(rng.integers(177,193))
    crushed = bool(rng.random() < .23)
    outline = [
        (110-neck/2,26),(110+neck/2,26),(110+neck/2,62),
        (right-12,shoulder),(right,shoulder+25),
    ]
    outline.extend((right-rng.uniform(0,18 if crushed else 4),y) for y in range(150,350,25))
    outline.extend([(right-4,371),(right-20,383),(left+18,383),(left,369)])
    outline.extend((left+rng.uniform(0,17 if crushed else 3),y) for y in range(345,140,-25))
    outline.extend([(left,shoulder+25),(left+12,shoulder),(110-neck/2,62)])
    draw_mask.polygon(outline, fill=255)
    body_color = choose(rng, ((157,196,201),(112,180,146),(189,183,155),(180,205,223),(124,164,207)))
    image = masked_layer((width,height), mask, body_color, rng, alpha=int(rng.integers(165,225)))
    draw = ImageDraw.Draw(image)
    draw.line(outline+[outline[0]], fill=(*tint(body_color,.63),220), width=3)
    # Bottle ribs and transparent PET highlights.
    for y in range(125,367,int(rng.integers(22,33))):
        draw.arc((left+4,y-7,right-4,y+12),0,180,fill=(224,241,244,150),width=2)
        draw.arc((left+5,y-4,right-5,y+9),180,350,fill=(85,127,142,135),width=2)
    draw.line([(left+17,shoulder+28),(left+12,335)],fill=(240,251,252,155),width=5)
    if rng.random() < .9:
        add_band(image,mask,(left,163,right,267),rng,("FRESH","SPRING","WAVE","PURE"),choose(rng,PALETTE))
    cap = choose(rng,PALETTE)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((110-neck/2-3,17,110+neck/2+3,47),radius=5,fill=(*cap,255))
    draw_mask.rounded_rectangle((110-neck/2-3,17,110+neck/2+3,47),radius=5,fill=255)
    for x in range(int(110-neck/2),int(110+neck/2),5):
        draw.line((x,22,x,42),fill=(*tint(cap,.65),255),width=2)
    draw.ellipse((110-neck/2-1,15,110+neck/2+1,27),fill=(*tint(cap,1.18),255))
    draw_mask.ellipse((110-neck/2-1,15,110+neck/2+1,27),fill=255)
    return image, mask, {"shape":"crushed" if crushed else "ribbed", "material":"translucent_pet"}


def can(rng):
    width,height = 240,330
    left,right,top,bottom = 30,210,34,296
    mask = Image.new("L",(width,height))
    draw_mask = ImageDraw.Draw(mask)
    dented = rng.random() < .3
    outline=[(left+10,top),(right-10,top),(right,top+24)]
    outline.extend((right-rng.uniform(0,19 if dented else 2),y) for y in range(80,272,24))
    outline.extend([(right-5,bottom),(right-24,bottom+12),(left+24,bottom+12),(left+4,bottom)])
    outline.extend((left+rng.uniform(0,18 if dented else 2),y) for y in range(270,70,-24))
    outline.append((left,top+22))
    draw_mask.polygon(outline,fill=255)
    base = choose(rng,PALETTE)
    image = masked_layer((width,height),mask,base,rng,metallic=True)
    add_band(image,mask,(left,87,right,243),rng,("FIZZ","ZEST","POP","SODA","SPARK"),base)
    draw = ImageDraw.Draw(image)
    draw.ellipse((left,17,right,68),fill=(120,130,132,255),outline=(220,225,221,255),width=5)
    draw_mask.ellipse((left,17,right,68),fill=255)
    draw.ellipse((left+12,24,right-12,60),fill=(176,183,182,255),outline=(80,90,91,255),width=2)
    draw.ellipse((101,37,142,53),fill=(56,67,68,255))
    draw.rounded_rectangle((89,26,122,45),radius=9,fill=(213,217,211,255),outline=(94,104,104,255),width=3)
    draw.ellipse((98,30,115,38),fill=(132,143,142,255))
    draw.arc((left+5,bottom-18,right-5,bottom+10),0,180,fill=(223,227,222,255),width=5)
    if dented:
        for _ in range(3):
            y=int(rng.integers(85,270))
            draw.line([(left+10,y),(120,y+rng.integers(-20,20)),(right-12,y+3)],fill=(215,213,207,95),width=2)
    return image,mask,{"shape":"dented" if dented else "cylinder", "material":"printed_aluminum"}


def wrapper(rng):
    width,height=340,250
    mask=Image.new("L",(width,height))
    draw_mask=ImageDraw.Draw(mask)
    left,right=25,315
    ytop=int(rng.integers(32,57)); ybottom=int(rng.integers(190,219))
    outline=[]
    for x in range(left,right+1,10):
        outline.append((x,ytop+rng.uniform(-8,8)))
    for y in range(ytop,ybottom,8):
        outline.append((right+rng.uniform(-9,2),y))
    for x in range(right,left-1,-10):
        outline.append((x,ybottom+rng.uniform(-8,8)))
    for y in range(ybottom,ytop,-8):
        outline.append((left+rng.uniform(-2,9),y))
    draw_mask.polygon(outline,fill=255)
    color=choose(rng,PALETTE)
    image=masked_layer((width,height),mask,color,rng,metallic=True)
    draw=ImageDraw.Draw(image)
    # Crimped seals and many folds, with class-independent package colors.
    for x0,x1 in ((left,left+27),(right-27,right)):
        draw.polygon([(x0,ytop+3),(x1,ytop+6),(x1,ybottom-3),(x0,ybottom-2)],fill=(*tint(color,.63),255))
        for x in range(x0+3,x1,5):
            draw.line((x,ytop+8,x+2,ybottom-7),fill=(*tint(color,1.28),230),width=2)
    center=(int(rng.integers(143,197)),int(rng.integers(100,149)))
    for _ in range(18):
        edge=choose(rng,outline)
        endpoint=(center[0]+rng.uniform(-65,65),center[1]+rng.uniform(-48,48))
        triangle=[edge,(edge[0]+rng.uniform(2,18),edge[1]+rng.uniform(-8,8)),endpoint]
        draw.polygon(triangle,fill=(*tint(color,rng.uniform(.75,1.35)),255))
        draw.line((edge,endpoint),fill=(229,230,222,int(rng.integers(70,150))),width=1)
    label_color=choose(rng,PALETTE)
    draw.rounded_rectangle((87,83,256,163),radius=18,fill=(*label_color,245))
    text=choose(rng,("SNACK","CRUNCH","OAT BITES","CRISPS","CHOCO"))
    ink=(245,242,224,255) if sum(label_color)<420 else (35,43,47,255)
    draw.text((171,111),text,font=font(23,True),fill=ink,anchor="mm")
    draw.text((171,143),choose(rng,("CLASSIC","ORIGINAL","ROASTED")),font=font(11),fill=ink,anchor="mm")
    # The procedural foil facets must never extend beyond the actual wrapper.
    rgba=np.array(image)
    rgba[...,3]=np.minimum(rgba[...,3],np.asarray(mask))
    return Image.fromarray(rgba),mask,{"shape":"creased_pouch", "material":"printed_foil_film"}


RENDERERS=(bottle,can,wrapper)


def background(rng,size):
    y,x=np.mgrid[0:size,0:size].astype(float)
    kind=choose(rng,("wood","concrete","matte","tile","kraft"))
    if kind=="wood":
        base=np.array(choose(rng,((151,114,77),(189,161,119),(93,71,52),(178,144,106))),float)
        grain=5*np.sin(y*.26+np.sin(x*.012)*4)+2*np.sin(y*1.1+x*.005)
        grain+=3*np.sin(y*.056+x*.012)
        texture=grain+rng.normal(0,2,(size,size))
        texture[(y.astype(int)%int(rng.integers(130,230)))<2]-=16
    elif kind=="tile":
        base=np.array(choose(rng,((207,207,198),(138,154,157),(174,163,147))),float)
        angle=rng.uniform(-.4,.4)
        u=x*math.cos(angle)+y*math.sin(angle);v=-x*math.sin(angle)+y*math.cos(angle)
        tile_size=int(rng.integers(120,210))
        texture=rng.normal(0,1.3,(size,size))
        texture[(np.mod(u,tile_size)<3)|(np.mod(v,tile_size)<3)]-=24
    else:
        bases={"concrete":((129,130,124),(182,180,172),(92,99,103)),
               "matte":((49,60,63),(194,206,204),(174,185,148),(105,98,120)),
               "kraft":((163,139,101),(194,173,133),(134,113,84))}
        base=np.array(choose(rng,bases[kind]),float)
        texture=rng.normal(0,3 if kind!="matte" else 1,(size,size))
        texture+=np.sin(x*.04+y*.012)*1.2
    light=1+rng.uniform(-.2,.2)*(x/size-.5)+rng.uniform(-.2,.2)*(y/size-.5)
    rgb=base[None,None,:]*light[...,None]+texture[...,None]
    return Image.fromarray(np.clip(rgb,0,255).astype(np.uint8)),kind


def transform(sprite,mask,rng,size):
    # Separate nearest-neighbor instance geometry from antialiased RGB edges.
    box=mask.getbbox()
    sprite=sprite.crop(box);mask=mask.crop(box)
    scale=rng.uniform(.45,1.03)*size/max(sprite.size)
    stretch=rng.uniform(.82,1.17)
    target=(max(24,int(sprite.width*scale*stretch)),max(24,int(sprite.height*scale)))
    sprite=sprite.resize(target,Image.Resampling.LANCZOS)
    mask=mask.resize(target,Image.Resampling.NEAREST)
    angle=float(rng.uniform(-180,180))
    sprite=sprite.rotate(angle,Image.Resampling.BICUBIC,expand=True)
    mask=mask.rotate(angle,Image.Resampling.NEAREST,expand=True)
    crop=mask.getbbox()
    return sprite.crop(crop),mask.crop(crop),{"angle_degrees":round(angle,3),"stretch":round(stretch,4)}


def bbox_overlap(candidate,previous):
    x0,y0,x1,y1=candidate
    for a,b,c,d in previous:
        area=max(0,min(x1,c)-max(x0,a))*max(0,min(y1,d)-max(y0,b))
        fraction=area/max(1,min((x1-x0)*(y1-y0),(c-a)*(d-b)))
        if fraction>.30:
            return True
    return False


def place(sprite,mask,rng,size,previous):
    # Clip occasional objects at an image edge to exercise YOLO normalization.
    max_width=int(size*.67);max_height=int(size*.73)
    ratio=min(1,max_width/sprite.width,max_height/sprite.height)
    if ratio<1:
        target=(max(1,int(sprite.width*ratio)),max(1,int(sprite.height*ratio)))
        sprite=sprite.resize(target,Image.Resampling.LANCZOS)
        mask=mask.resize(target,Image.Resampling.NEAREST)
    for attempt in range(90):
        margin=-int(min(sprite.size)*.12) if rng.random()<.13 else 15
        x=int(rng.integers(margin,max(margin+1,size-sprite.width-margin)))
        y=int(rng.integers(margin,max(margin+1,size-sprite.height-margin)))
        candidate=(x,y,x+sprite.width,y+sprite.height)
        if not bbox_overlap(candidate,previous):
            return sprite,mask,x,y
    # Shrink instead of hiding an earlier instance or silently losing a label.
    target=(max(24,int(sprite.width*.7)),max(24,int(sprite.height*.7)))
    return place(sprite.resize(target,Image.Resampling.LANCZOS),
                 mask.resize(target,Image.Resampling.NEAREST),rng,size,previous)


def scene(seed,size,classes):
    rng=np.random.default_rng(seed)
    canvas,kind=background(rng,size)
    canvas=canvas.convert("RGBA")
    instance_mask=np.zeros((size,size),dtype=np.uint8)
    previous=[]; specs=[]
    for instance_id,class_id in enumerate(classes,start=1):
        sprite,mask,appearance=RENDERERS[class_id](rng)
        sprite,mask,pose=transform(sprite,mask,rng,size)
        sprite,mask,x,y=place(sprite,mask,rng,size,previous)
        previous.append((x,y,x+sprite.width,y+sprite.height))
        fullmask=Image.new("L",(size,size));fullmask.paste(mask,(x,y))
        shadow_mask=Image.new("L",(size,size))
        shadow_mask.paste(mask,(x+int(rng.integers(5,14)),y+int(rng.integers(8,21))))
        shadow_mask=shadow_mask.filter(ImageFilter.GaussianBlur(float(rng.uniform(4,9))))
        shadow=Image.new("RGBA",(size,size),(18,15,12,0))
        shadow.putalpha(shadow_mask.point(lambda a:int(a*.22)))
        canvas.alpha_composite(shadow)
        layer=Image.new("RGBA",(size,size));layer.paste(sprite,(x,y))
        canvas.alpha_composite(layer)
        selected=np.asarray(fullmask)>127
        instance_mask[selected]=instance_id
        specs.append({"instance_id":instance_id,"class_id":class_id,**appearance,**pose,
                      "unoccluded_pixels":int(np.count_nonzero(selected))})
    # These effects do not change instance-mask geometry; masks exclude shadows.
    image=canvas.convert("RGB")
    blur=float(rng.uniform(0,.68)) if rng.random()<.42 else 0
    if blur:
        image=image.filter(ImageFilter.GaussianBlur(blur))
    values=np.asarray(image).astype(np.float32)
    gain=float(rng.uniform(.82,1.16));noise=float(rng.uniform(.5,3.2))
    values=values*gain+rng.normal(0,noise,values.shape)
    image=Image.fromarray(np.clip(values,0,255).astype(np.uint8))
    objects=[]
    for spec in specs:
        ys,xs=np.where(instance_mask==spec["instance_id"])
        if not len(xs):
            raise RuntimeError("An instance was fully occluded; scene must be regenerated")
        box=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]
        objects.append({**spec,"bbox_xyxy":box,"visible_pixels":len(xs),
                        "visible_fraction":round(len(xs)/spec["unoccluded_pixels"],5)})
    return image,Image.fromarray(instance_mask),objects,{"background":kind,"brightness":round(gain,3),
                                                       "blur_radius":round(blur,3),"noise_sigma":round(noise,3)}


def yolo_row(obj,width,height):
    x0,y0,x1,y1=obj["bbox_xyxy"]
    return f"{obj['class_id']} {(x0+x1)/(2*width):.8f} {(y0+y1)/(2*height):.8f} {(x1-x0)/width:.8f} {(y1-y0)/height:.8f}"


def annotated_tile(root,row,tile_size=320):
    image=Image.open(root/row["image"]).convert("RGB")
    draw=ImageDraw.Draw(image)
    colors=((73,211,125),(242,114,75),(85,152,239))
    for obj in row["objects"]:
        box=obj["bbox_xyxy"];color=colors[obj["class_id"]]
        draw.rectangle((box[0],box[1],box[2]-1,box[3]-1),outline=color,width=4)
        text=CLASSES[obj["class_id"]]
        tx,ty=box[0],max(0,box[1]-23)
        draw.rectangle((tx,ty,tx+len(text)*10+8,ty+23),fill=(17,26,31))
        draw.text((tx+4,ty+3),text,font=font(15,True),fill=color)
    image=image.resize((tile_size,tile_size),Image.Resampling.LANCZOS)
    tile=Image.new("RGB",(tile_size,tile_size+30),(20,29,35))
    tile.paste(image,(0,0))
    caption=f"{row['split']} / {len(row['objects'])} objects / {row['seed']}"
    ImageDraw.Draw(tile).text((9,tile_size+7),caption,font=font(13),fill=(223,230,233))
    return tile


def preview_sheet(root,rows):
    # One negative plus different foreground-count/class examples from each split.
    picked=[]
    for split in ("train","val","test"):
        group=[r for r in rows if r["split"]==split]
        picked.append(next(r for r in group if not r["objects"]))
        for class_id in range(3):
            picked.append(next(r for r in group if r["objects"] and r["objects"][0]["class_id"]==class_id))
    sheet=Image.new("RGB",(1280,1120),(15,24,29))
    draw=ImageDraw.Draw(sheet)
    draw.text((22,15),"ECOSORT  /  SYNTHETIC YOLO TEST DATA",font=font(25,True),fill=(237,241,237))
    draw.text((22,47),"Procedural renders | exact visible-instance boxes | not real-world validation",font=font(16),fill=(161,181,184))
    for index,row in enumerate(picked):
        sheet.paste(annotated_tile(root,row),(index%4*320,70+index//4*350))
    previews=root/"previews";previews.mkdir()
    sheet.save(previews/"contact_sheet.jpg",quality=92)
    # Also include an uncluttered three-object scene at full resolution.
    multi=next((r for r in rows if len(r["objects"])==3),None)
    if multi:
        annotated_tile(root,multi,640).save(previews/"three_objects.jpg",quality=94)


def build(output,counts,size,seed,negative_fraction,make_zip):
    output=output.resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite {output}. Choose a new --output folder.")
    output.mkdir(parents=True)
    names="\n".join(f"  {i}: {name}" for i,name in enumerate(CLASSES))
    (output/"data.yaml").write_text(
        "# Relative paths resolve from this YAML file's directory.\n"
        "# Synthetic testing only: validate deployed models on real held-out photos.\n"
        "train: images/train\nval: images/val\ntest: images/test\nnc: 3\nnames:\n"+names+"\n",
        encoding="utf-8",
    )
    rows=[];summary={}
    with (output/"manifest.jsonl").open("w",encoding="utf-8") as manifest:
        for split_id,(split,count) in enumerate(counts.items()):
            for folder in ("images","labels","masks"):
                (output/folder/split).mkdir(parents=True)
            split_rng=np.random.default_rng(seed+split_id*1_000_000+999_999)
            negative_count=max(1,round(count*negative_fraction))
            negative_indices=set(int(v) for v in split_rng.choice(count,negative_count,replace=False))
            stats=Counter();primary_index=0
            for index in range(count):
                scene_seed=seed+split_id*1_000_000+index
                if index in negative_indices:
                    targets=[]
                else:
                    primary=primary_index%3;primary_index+=1
                    targets=[primary]
                    n=int(split_rng.choice((1,2,3),p=(.48,.37,.15)))
                    targets.extend(int(v) for v in split_rng.integers(0,3,size=n-1))
                image,mask,objects,render_info=scene(scene_seed,size,targets)
                stem=f"{split}_{index:05d}"
                row={"image":f"images/{split}/{stem}.jpg","label":f"labels/{split}/{stem}.txt",
                     "mask":f"masks/{split}/{stem}.png","split":split,"seed":scene_seed,
                     "width":size,"height":size,"synthetic":True,"objects":objects,**render_info}
                image.save(output/row["image"],quality=int(split_rng.integers(85,96)),subsampling=0)
                mask.save(output/row["mask"],optimize=True)
                text="\n".join(yolo_row(obj,size,size) for obj in objects)
                (output/row["label"]).write_text(text+("\n" if text else ""),encoding="utf-8")
                row["sha256"]=hashlib.sha256((output/row["image"]).read_bytes()).hexdigest()
                manifest.write(json.dumps(row)+"\n")
                rows.append(row)
                stats.update(CLASSES[obj["class_id"]] for obj in objects)
                if (index+1)%100==0 or index+1==count:
                    print(f"{split}: {index+1}/{count}",flush=True)
            summary[split]={"images":count,"negative_images":negative_count,"instances":dict(stats)}
    preview_sheet(output,rows)
    report={"dataset":"EcoSort synthetic YOLO test v1","created_utc":datetime.now(timezone.utc).isoformat(),
            "classes":dict(enumerate(CLASSES)),"image_size":[size,size],"seed":seed,"splits":summary,
            "total_images":len(rows),"generator":"procedural NumPy/Pillow 2.5D object sprites",
            "annotation":"tight final visible-instance mask bounds; pixel max edges are exclusive; shadows excluded",
            "source_images":"none; every scene is procedurally generated",
            "limitation":"Pipeline and synthetic-scene testing only. No real-world waste accuracy claim.",
            "versions":{"numpy":np.__version__,"pillow":Image.__version__}}
    (output/"dataset_summary.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    (output/"README.md").write_text(
        "# EcoSort synthetic YOLO test dataset\n\n"
        f"{len(rows)} procedurally rendered {size}x{size} images. Classes: 0 plastic_bottle, 1 metal_can, 2 wrapper.\n\n"
        "This dataset is ready for YOLO object-detection pipeline tests, not real-world accuracy claims. "
        "The bottle, can, and wrapper appearances are synthetic; collect and label real images before deploying.\n\n"
        "- `data.yaml`: portable Ultralytics dataset configuration.\n"
        "- `images/` and `labels/`: paired JPG / YOLO txt files, split into train, val, test.\n"
        "- `masks/`: per-image 8-bit instance masks (0 background, 1..N objects).\n"
        "- `manifest.jsonl`: class/instance IDs, exact pixel boxes, seeds and rendering settings.\n"
        "- `dataset_summary.json`: split counts, class counts and provenance.\n"
        "- `previews/contact_sheet.jpg`: annotated examples for visual checking.\n\n"
        "YOLO rows are `class_id center_x center_y width height`, normalized to [0,1]. "
        "Empty scenes have empty label files; there is no `other` detector class. "
        "Visible-mask boxes include the object's visible surface, not its drop shadow.\n\n"
        "The existing EcoSort MobileNet classifier and RUN_TRAINING.cmd use a different format. "
        "Use the separate YOLO training instructions in the companion yolo_synthetic folder.\n\n"
        "## Optional training smoke test\n\n"
        "With Ultralytics installed in a separate Python environment, run this command from "
        "the folder containing this data.yaml:\n\n"
        "```text\nyolo detect train model=yolo11n.pt data=data.yaml epochs=3 imgsz=640 batch=4 device=cpu workers=0\n```\n\n"
        "This generator does not install Ultralytics or train weights. The command can download "
        "pretrained weights on first use. Three epochs test the pipeline, not real-world readiness.\n\n"
        "Format reference: https://docs.ultralytics.com/datasets/detect/\n",
        encoding="utf-8",
    )
    if make_zip:
        archive=output.with_suffix(".zip")
        if archive.exists():
            raise SystemExit(f"Dataset generated, but refusing to overwrite archive {archive}")
        with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED,compresslevel=3) as package:
            for path in sorted(output.rglob("*")):
                if path.is_file():
                    package.write(path,Path(output.name)/path.relative_to(output))
        print(f"ZIP: {archive}",flush=True)
    print(json.dumps(report,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=PROJECT/"datasets"/"trash_yolo_synthetic_v1")
    parser.add_argument("--train",type=int,default=960)
    parser.add_argument("--val",type=int,default=120)
    parser.add_argument("--test",type=int,default=120)
    parser.add_argument("--size",type=int,default=640)
    parser.add_argument("--seed",type=int,default=930909)
    parser.add_argument("--negative-fraction",type=float,default=.125)
    parser.add_argument("--zip",action="store_true",dest="make_zip")
    args=parser.parse_args()
    if min(args.train,args.val,args.test)<12:
        parser.error("each split needs at least 12 images for coverage and previews")
    if not 256<=args.size<=1280:
        parser.error("--size must be between 256 and 1280")
    if not .05<=args.negative_fraction<=.4:
        parser.error("--negative-fraction must be between .05 and .4")
    if min(args.seed,args.train,args.val,args.test)<0 or max(args.train,args.val,args.test)>=999_000:
        parser.error("invalid seed/count; each split must contain fewer than 999000 images")
    build(args.output,{"train":args.train,"val":args.val,"test":args.test},args.size,
          args.seed,args.negative_fraction,args.make_zip)


if __name__=="__main__":
    main()
