#include <gazebo/transport/transport.hh>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/gazebo_client.hh>

#include <iostream>
//typedef const boost::shared_ptr<const gazebo::msgs::Pose> PosePtr;

void cb(ConstImagesStampedPtr &imagesStamped)
{
  //std::cout << imagesStamped->DebugString();
  ::google::protobuf::int32 sec = imagesStamped->time().sec();
  ::google::protobuf::int32 nsec = imagesStamped->time().nsec();
  std::cout << "Read time: sec: " << sec << " nsec: " << nsec << std::endl;

  for (int i =0; i < imagesStamped->image_size(); ++i)
  {
    const ::gazebo::msgs::Image &image = imagesStamped->image(i);
    ::google::protobuf::uint32 width = image.width();
    std::cout<<"image width: "<<width<<std::endl;
  }
}

/////////////////////////////////////////////////
int main(int _argc, char **_argv)
{
  gazebo::client::setup(_argc, _argv);
  gazebo::transport::NodePtr node;
  node = gazebo::transport::NodePtr(new gazebo::transport::Node());
  node->Init();
  gazebo::transport::SubscriberPtr sub = node->Subscribe("~/camera/link/camera/image", cb);
  std::cout<<"subscribe success"<<std::endl;

  while (true)
    {
      //std::cout<<"loop entered"<<std::endl;
      gazebo::common::Time::MSleep(10);
    }

  gazebo::client::shutdown();
}
