package main

import (
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"io/ioutil"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"sort"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/nfnt/resize"
	"github.com/owulveryck/onnx-go"
	"github.com/owulveryck/onnx-go/backend/x/gorgonnx"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/native"
)

const (
	confidenceThreshold = 0.9
	classProbaThreshold = 0.3
	hSize, wSize  = 416, 416
	blockSize     = 32
	gridHeight    = 13
	gridWidth     = 13
	boxesPerCell  = 5
	numClasses    = 20
	envConfPrefix = "yolo"
)

var (
	anchors = []float64{1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52}
	classes = []string{"aeroplane", "bicycle", "bird", "boat", "bottle",
		"bus", "car", "cat", "chair", "cow",
		"diningtable", "dog", "horse", "motorbike", "person",
		"pottedplant", "sheep", "sofa", "train", "tv/monitor"}
	scaleFactor = float32(1) // The scale factor to resize the image to hSize*wSize
)

type element struct {
	prob  float64
	class string
}

type byProba []element

func (b byProba) Len() int           { return len(b) }
func (b byProba) Swap(i, j int)      { b[i], b[j] = b[j], b[i] }
func (b byProba) Less(i, j int) bool { return b[i].prob < b[j].prob }

type box struct {
	r          image.Rectangle
	gridcell   []int
	confidence float64
	classes    []element
}

type byConfidence []box

func (b byConfidence) Len() int           { return len(b) }
func (b byConfidence) Swap(i, j int)      { b[i], b[j] = b[j], b[i] }
func (b byConfidence) Less(i, j int) bool { return b[i].confidence < b[j].confidence }


func main() {
	backend := gorgonnx.NewGraph()
	model := onnx.NewModel(backend)

	b, _ := ioutil.ReadFile("/mnt/c/Users/Peter/Projects/yolo_server_go/tinyyolov2-8.onnx")
	err := model.UnmarshalBinary(b)
	if err != nil {
		fmt.Println(`Encountered error: %v`, err)
	}

	f, err := os.Open("/mnt/c/Users/Peter/Projects/yolo_server_go/dog.jpg")
	if err != nil {
		log.Fatal(err)
	}
	img, err := jpeg.Decode(f)
	if err != nil {
		log.Fatal(err)
	}

	imgRescaled := image.NewNRGBA(image.Rect(0, 0, wSize, hSize))
	color := color.RGBA{0, 0, 0, 255}

	draw.Draw(imgRescaled, imgRescaled.Bounds(), &image.Uniform{color}, image.ZP, draw.Src)
	var m image.Image
	if (img.Bounds().Max.X - img.Bounds().Min.X) > (img.Bounds().Max.Y - img.Bounds().Min.Y) {
		scaleFactor = float32(img.Bounds().Max.Y-img.Bounds().Min.Y) / float32(hSize)
		m = resize.Resize(0, hSize, img, resize.Lanczos3)
	} else {
		scaleFactor = float32(img.Bounds().Max.X-img.Bounds().Min.X) / float32(wSize)
		m = resize.Resize(wSize, 0, img, resize.Lanczos3)
	}

	switch m.(type) {
	case *image.NRGBA:
			draw.Draw(imgRescaled, imgRescaled.Bounds(), m.(*image.NRGBA), image.ZP, draw.Src)
	case *image.YCbCr:
		draw.Draw(imgRescaled, imgRescaled.Bounds(), m.(*image.YCbCr), image.ZP, draw.Src)
	default:
		log.Fatal("unhandled type")
	}
	inputT := tensor.New(tensor.WithShape(1, 3, hSize, wSize), tensor.Of(tensor.Float32))
	err = ImageToBCHW(imgRescaled, inputT)
	if err != nil {
		log.Fatal(err)
	}
	model.SetInput(0, inputT)
	
	
	err = backend.Run()
	if err != nil {
		fmt.Println(err)
	}

	output, err := model.GetOutputTensors()
	dense := output[0].(*tensor.Dense)
	err = dense.Reshape(125, 13, 13)
	data, err := native.Tensor3F32(dense)
	if err != nil {
		fmt.Println(err)
		log.Fatal(err)
	}

	fmt.Println("Calculating highest prob boxes...")
	var boxes = make([]box, gridHeight*gridWidth*boxesPerCell)
	var counter int
	for cx := 0; cx < gridWidth; cx++ {
		for cy := 0; cy < gridHeight; cy++ {
			for b := 0; b < boxesPerCell; b++ {
				channel := b * (numClasses + 5)
				tx := data[channel][cx][cy]
				ty := data[channel+1][cx][cy]
				tw := data[channel+2][cx][cy]
				th := data[channel+3][cx][cy]
				tc := data[channel+4][cx][cy]
				tclasses := make([]float32, 20)
				for i := 0; i < 20; i++ {
					tclasses[i] = data[channel+5+i][cx][cy]
				}
				// The predicted tx and ty coordinates are relative to the location
				// of the grid cell; we use the logistic sigmoid to constrain these
				// coordinates to the range 0 - 1. Then we add the cell coordinates
				// (0-12) and multiply by the number of pixels per grid cell (32).
				// Now x and y represent center of the bounding box in the original
				// 416x416 image space.
				// https://github.com/hollance/Forge/blob/04109c856237faec87deecb55126d4a20fa4f59b/Examples/YOLO/YOLO/YOLO.swift#L154
				x := int((float32(cx) + sigmoid(tx)) * blockSize)
				y := int((float32(cy) + sigmoid(ty)) * blockSize)
				// The size of the bounding box, tw and th, is predicted relative to
				// the size of an "anchor" box. Here we also transform the width and
				// height into the original 416x416 image space.
				w := int(exp(tw) * anchors[2*b] * blockSize)
				h := int(exp(th) * anchors[2*b+1] * blockSize)

				boxes[counter] = box{
					gridcell:   []int{cx, cy},
					r:          image.Rect(max(y-w/2, 0), max(x-h/2, 0), min(y+w/2, wSize), min(x+h/2, hSize)),
					confidence: sigmoid64(tc),
					classes:    getOrderedElements(softmax(tclasses)),
				}
				counter++
			}
		}
	}

	boxes = sanitize(boxes)
	printClassification(boxes)
	

	/*
	router := gin.Default()
	router.POST("/upload", saveFileHandler)

	router.Run("localhost:8080")
	*/
}

func printClassification(boxes []box) {
	var elements []element
	for _, box := range boxes {
		if box.classes[0].prob > confidenceThreshold {
			elements = append(elements, box.classes...)
			fmt.Printf("at (%v) with confidence %2.2f%%: %v\n", box.r, box.confidence, box.classes[:3])
		}
	}
	sort.Sort(sort.Reverse(byProba(elements)))
	for _, c := range elements {
		if c.prob > 0.4 {
			fmt.Println(c)
		}
	}

}

func getOrderedElements(input []float64) []element {
	elems := make([]element, len(input))
	for i := 0; i < len(elems); i++ {
		elems[i] = element{
			prob:  input[i],
			class: classes[i],
		}
	}
	sort.Sort(sort.Reverse(byProba(elems)))
	return elems
}

// from https://medium.com/@jonathan_hui/real-time-object-detection-with-yolo-yolov2-28b1b93e2088
// 1- Sort the predictions by the confidence scores.
// 2- Start from the top scores, ignore any current prediction if we find any previous predictions that have the same class and IoU > 0.5 with the current prediction.
// 3- Repeat step 2 until all predictions are checked.
func sanitize(boxes []box) []box {
	sort.Sort(sort.Reverse(byConfidence(boxes)))

	for i := 1; i < len(boxes); i++ {
		if boxes[i].confidence < confidenceThreshold {
			boxes = boxes[:i]
			break
		}
		if boxes[i].classes[0].prob < classProbaThreshold {
			boxes = boxes[:i]
			break
		}
		for j := i + 1; j < len(boxes); {
			iou := iou(boxes[i].r, boxes[j].r)
			if iou > 0.5 && boxes[i].classes[0].class == boxes[j].classes[0].class {
				boxes = append(boxes[:j], boxes[j+1:]...)
				continue
			}
			j++
		}
	}
	return boxes
}

func sigmoid(sum float32) float32 {
	return float32(1.0 / (1.0 + math.Exp(float64(-sum))))
}

func sigmoid64(sum float32) float64 {
	return 1.0 / (1.0 + math.Exp(float64(-sum)))
}

func exp(val float32) float64 {
	return math.Exp(float64(val))
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func softmax(a []float32) []float64 {
	var sum float64
	output := make([]float64, len(a))
	for i := 0; i < len(a); i++ {
		output[i] = math.Exp(float64(a[i]))
		sum += output[i]
	}
	for i := 0; i < len(output); i++ {
		output[i] = output[i] / sum
	}
	return output
}

func area(r image.Rectangle) int {
	return max(0, r.Max.X-r.Min.X-1) * max(0, r.Max.Y-r.Min.Y-1)
}

// evaluate the intersection over union of two rectangles
func iou(r1, r2 image.Rectangle) float64 {
	// get the intesection rectangle
	intersection := image.Rect(
		max(r1.Min.X, r2.Min.X),
		max(r1.Min.Y, r2.Min.Y),
		min(r1.Max.X, r2.Max.X),
		min(r1.Max.Y, r2.Max.Y),
	)
	// compute the area of intersection rectangle
	interArea := area(intersection)
	r1Area := area(r1)
	r2Area := area(r2)
	// compute the intersection over union by taking the intersection
	// area and dividing it by the sum of prediction + ground-truth
	// areas - the interesection area
	return float64(interArea) / float64(r1Area+r2Area-interArea)
}

// ImageToBCHW convert an image to a BCHW tensor
// this function returns an error if:
//
//   - dst is not a pointer
//   - dst's shape is not 4
//   - dst' second dimension is not 1
//   - dst's third dimension != i.Bounds().Dy()
//   - dst's fourth dimension != i.Bounds().Dx()
//   - dst's type is not float32 or float64 (temporary)
func ImageToBCHW(img image.Image, dst tensor.Tensor) error {
	w := img.Bounds().Dx()
	h := img.Bounds().Dy()
	err := verifyBCHWTensor(dst, h, w, false)
	if err != nil {
		return err
	}

	switch dst.Dtype() {
	case tensor.Float32:
		for x := 0; x < w; x++ {
			for y := 0; y < h; y++ {
				r, g, b, a := img.At(x, y).RGBA()
				if a != 65535 {
					return errors.New("transparency not handled")
				}
				err := dst.SetAt(float32(uint8(r/0x101)), 0, 0, y, x)
				if err != nil {
					return err
				}
				err = dst.SetAt(float32(uint8(g/0x101)), 0, 1, y, x)
				if err != nil {
					return err
				}
				err = dst.SetAt(float32(uint8(b/0x101)), 0, 2, y, x)
				if err != nil {
					return err
				}
			}
		}
	default:
		return fmt.Errorf("%v not handled yet", dst.Dtype())
	}
	return nil
}

func verifyBCHWTensor(dst tensor.Tensor, h, w int, cowardMode bool) error {
	// check if tensor is a pointer
	rv := reflect.ValueOf(dst)
	if rv.Kind() != reflect.Ptr || rv.IsNil() {
		return errors.New("cannot decode image into a non pointer or a nil receiver")
	}
	// check if tensor is compatible with BCHW (4 dimensions)
	if len(dst.Shape()) != 4 {
		return fmt.Errorf("Expected a 4 dimension tensor, but receiver has only %v", len(dst.Shape()))
	}
	// Check the batch size
	if dst.Shape()[0] != 1 {
		return errors.New("only batch size of one is supported")
	}
	if cowardMode && dst.Shape()[1] != 1 {
		return errors.New("Cowardly refusing to insert a gray scale into a tensor with more than one channel")
	}
	if dst.Shape()[2] != h || dst.Shape()[3] != w {
		return fmt.Errorf("cannot fit image into tensor; image is %v*%v but tensor is %v*%v", h, w, dst.Shape()[2], dst.Shape()[3])
	}
	return nil
}

func saveFileHandler(c *gin.Context) {
	file, err := c.FormFile("file")

	// The file cannot be received
	if err != nil {
		fmt.Println(err)
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{
			"message": "No file received",
		})
		return
	}

	// Get file info
	extension := filepath.Ext(file.Filename)
	// Generate random file name for uploaded file
	newFileName := uuid.New().String() + extension

	// The file was received successfully so save it
	if err := c.SaveUploadedFile(file, "/tmp/yolo_server/" + newFileName); err != nil {
		fmt.Println(err)
		c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{
			"message": "Unable to save the file",
		})
		return
	}

	// File saved successfully. Return result
	c.IndentedJSON(http.StatusOK, gin.H{
		"message": "Your file has been successfully uploaded.",
	})
}
