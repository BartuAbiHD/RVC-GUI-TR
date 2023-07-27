<div align="center">

<h1>RVC GUI TURKISH<br><br>
  
Yalnızca ses dosyası çıkarımı için

  <br>

  

</div>

  

 

  
## GUI

![GUI](https://raw.githubusercontent.com/BartuAbiHD/RVC-GUI-TR/main/docs/GUI.png)
 <br><br>
  
## Windows kullanıcıları için doğrudan kurulum
## [Windows-pkg](https://github.com/BartuAbiHD/RVC-GUI-TR/releases/tag/Windows-pkg)
  
<br><br>
## Çevreyi Hazırlamak


* Yapmadıysanız Python + 3.8 sürümünü yükleyin:

* Bu komutları yürütün

Nvıdıa kartlı Windows'lar
```bash
python -m pip install -U pip setuptools wheel
pip install -U torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
Diğer
```
python -m pip install -U pip setuptools wheel
pip install -U torch torchaudio 
pip install -r requirements.txt
```

Apple silikon Mac'leri düzeltme
```
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu

export PYTORCH_ENABLE_MPS_FALLBACK=1
```
<br>

* [hubert_base.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt/)'i indirin ve kök klasörüne yerleştirin

<br>
 
* Ardından RVC GUI'yi başlatmak için bu komutu kullanın:
```bash
python rvcgui.py
```
Veya bu dosyayı Windows'ta çalıştırın
```
RVC-GUI.bat
```

# Modelleri yükleme
Bir Modeli Zip dosyasından içe aktarmak için İçe Aktar düğmesini kullanın, 
* .Zip ".pth" ağırlık dosyasını içermelidir. 
* .Zip'in ".index" özellik alma dosyalarını içermesi önerilir.

Veya modeli manuel olarak kök dizindeki "modeller" klasörüne yerleştirin.
```
modeller
├───Kişi1
│   ├───xxxx.pth
│   ├───xxxx.index
│   └───xxxx.npy
└───Kişi2
    ├───xxxx.pth
    ├───...
    └───...
````
<br>


<br> 

### Modeller nasıl alınır?
* [ Trias AI](https://discord.gg/tpy6JbZhh8) Discord sunucusuna katılın. 
* [HuggingFace](https://huggingface.co/BartuAbiHD/rvc/tree/main) üzerinde topluluk modelleri için [burayı](https://huggingface.co/BartuAbiHD/rvc/tree/main) ziyaret edebilirsiniz. 
