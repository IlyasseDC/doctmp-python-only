# MODIFICATIONS.md

## Modifications made to the original code to make it work

---

### `eval_dtd.py`

#### 1. Ensure CPU-only execution
- **Changed:**
  ```python
  model = seg_dtd('',2).cuda()
- **to:**
device = torch.device('cpu')
model = seg_dtd("",2).to(device)
model = nn.DataParallel(model)
- **Reason: to force model on CPU, matching same setup as SROIE inference**

#### 2. Fix DataLoader workers on Windows
- **Changed:**
  train_loader1 = DataLoader(dataset=test_data, batch_size=6, num_workers=12, shuffle=False)
- **to:**
train_loader1 = DataLoader(dataset=test_data, batch_size=6, num_workers=0, shuffle=False)
- **Reason: avoids Windows multiprocessing pickling issue with LMDB**

#### 3. Convert DCT to long for embedding
- **add:**
    dct_coef = dct_coef.long()
    data, target, dct_coef, qs = Variable(data.to(device)), Variable(target.to(device)), Variable(dct_coef.to(device)), Variable(qs.unsqueeze(1).to(device))
- **Reason: ensure DCT indices are integers for torch Embedding**

#### 4. Fix NamedTemporaryFile for Windows
- **Changed:**
    with tempfile.NamedTemporaryFile(delete=True) as tmp:
    im.save(tmp, "JPEG", quality=q)
    im_array = cv2.imread(tmp.name, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
- **to:**
   with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
    tmp_name = tmp.name
    im.save(tmp_name, "JPEG", quality=q)
    im_array = cv2.imread(tmp_name, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    os.remove(tmp_name)
- **Reason: avoid Windows file lock that prevents OpenCV from reading the temp file**

### `dtd.py`

#### 1. Fix library import
- **add to imports:**
import sys
sys.modules['dtd'] = sys.modules[__name__]


#### `venv\Lib\site-packages\torch\nn\modules\activation`
- **Changed:**
line 734: gelu(input, approximate=self.approximate)
- **to:**
gelu(input)
- **Reason: fix the issues of compatibility in the newer version of pytroch**