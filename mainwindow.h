#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>
#include <QFileDialog>
#include <QTextCodec>
#include <QWidget>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
using namespace cv;
#include<string>
#include<iostream>
using namespace std;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    void image_resize(Mat &image,Mat &image_new);
    void image_toGray(Mat &image,Mat &image_gray);
    void image_calEnergy(Mat &image,Mat &G);
    void image_calEnergyAll(const Mat &G,Mat &M,Mat &R);
    void image_findEnergyLine(Mat &image,Mat &M,const Mat &R);
    void image_showEnergyLine(Mat &image,Mat &MinEnergyLine);
    void image_deleteEnergyLine(const Mat &image,Mat &image_new,const Mat &MinEnergyLine);


private:
    Ui::MainWindow *ui;
    Mat image;
    Mat image_dst;
    Mat image_new;
    Mat image_src;
    string name;

    //static const int SIZE = 50;
    int SIZE;

private slots:
    void OpenImageClicked();
    void ProcessClicked();
};

#endif // MAINWINDOW_H
