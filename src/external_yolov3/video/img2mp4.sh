cd ~
mkdir data -p
cd data
mkdir img_temp
mv ~/.ros/frame*.jpg ~/data/img_temp/

cd ~/data/img_temp
ffmpeg -framerate 15 -i frame%05d.jpg -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output.mp4



