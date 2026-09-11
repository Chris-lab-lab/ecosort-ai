# EcoSort V2 AI architecture

The deployed model remains small and fully integer for the NXP i.MX93. The improvement is not a larger detector: it separates **what material looks most likely** from **whether the image is safe to route at all**.

## Training and export

```mermaid
flowchart LR
    D[Local photos] --> M[Reviewed class folders]
    T[TACO crops] --> M
    C[Saved camera corrections] --> M
    M --> A[Lighting, translation, crop<br/>and contrast augmentation]
    A --> B[MobileNetV2 0.35<br/>feature extractor]
    B --> E[64-value embedding]
    E --> H1[Material head<br/>plastic / general-paper / metal]
    E --> H2[Validity head<br/>supported / reject]
    E --> P[Per-material prototypes<br/>and cosine-distance limits]
    H2 --> V[Calibrate validity threshold<br/>on validation data]
    H1 --> Q[Full INT8 export]
    H2 --> Q
    E --> Q
    P --> J[open_set.json]
    V --> J
```

The `other` images supervise the validity head. They do not become a fourth material competing with plastic, metal, and the middle-bin class. This makes rejection a separate decision and allows the unknown set to grow without changing the three physical routes.

## Live decision path

```mermaid
flowchart LR
    CAM[USB camera] --> ROI[Centered one-item ROI]
    ROI --> INT8[INT8 TFLite / Ethos-U]
    INT8 --> MAT[Material probabilities]
    INT8 --> VAL[Supported probability]
    INT8 --> EMB[Feature embedding]
    MAT --> AVG[Multi-frame averaging]
    VAL --> AVG
    EMB --> DIST[Cosine distance to<br/>winning material prototype]
    AVG --> GATE{All safety checks pass?}
    DIST --> GATE
    SENSOR[Optional digital<br/>metal sensor] --> GATE
    GATE -->|no| CLOSED[All lids closed]
    GATE -->|yes| ROUTE[One material route]
    ROUTE --> PCA[PCA9685]
    PCA --> LID[One SG90 lid]
```

The gate requires adequate material confidence and margin, a supported probability above the calibrated threshold, a feature distance within the learned class boundary, stable agreement across frames, and camera/sensor agreement when the sensor is enabled.

## Artifact contract

Keep these together when deploying an uncompiled model:

- `waste_classifier_int8.tflite`: material, validity, and embedding outputs.
- `labels.txt`: the three material outputs in exact tensor order.
- `open_set.json`: validity threshold, material prototypes, and distance limits.

When Vela writes the compiled model into a subfolder, pass the original sidecar explicitly with `--metadata artifacts/open_set.json`.
