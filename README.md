# SEM-SEG

Data tools final project. Segment silica spheres out from a SEM image.

## Setup

Frontend: Electron + Svelte + TailwindCss + FlowbiteUI
Backend: Python with tensorflow bundle

<https://electron-vite.org/guide/>
<https://flowbite-svelte.com/docs/pages/quickstart>
<https://segment-anything.com/demo>
<https://konvajs.org/docs/index.html>

## Steps w/ screenshots

Initialize electron-svelte project
Initialize UI library
Showcase python subprocess

<https://medium.com/red-buffer/integrating-python-flask-backend-with-electron-nodejs-frontend-8ac621d13f72>

## Development

Prerequisites:  

- Node.js w/ npm
- Python w/ venv & pip  
- VSCode

npm run dev: start hot frontend reload
npm run build: build python to exe and compile svelte/ts to pure js
npm run build:win: generate distributable files.

## Environment

pytorch for cpu
numpy
matplotlib
opencv-python
pandas
pycocotools

<https://maxjoas.medium.com/finetune-segment-anything-sam-for-images-with-multiple-masks-34514ee811bb>
