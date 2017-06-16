#include <gazebo/transport/transport.hh>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/gazebo_client.hh>

#include <iostream>
//typedef const boost::shared_ptr<const gazebo::msgs::Pose> PosePtr;

/////////////////////////////////////////////////
// Function is called everytime a message is received.
void cb(ConstPosesStampedPtr &posesStamped)
{
  //std::cout << posesStamped->DebugString();
  //std::cout<<"pose size"<<posesStamped->pose_size(); pose_size=8
  for (int i =0; i < posesStamped->pose_size(); ++i)
  {
    const ::gazebo::msgs::Pose &pose = posesStamped->pose(i);
    std::string name = pose.name();
    ::google::protobuf::uint32 id = pose.id();
    std::cout << "name:"<<name<<" id:"<<id<<std::endl;
    if (true)
    {
      const ::gazebo::msgs::Vector3d &position = pose.position();

      double x = position.x();
      double y = position.y();
      double z = position.z();

      std::cout << "Read position: x: " << x
          << " y: " << y << " z: " << z << std::endl;
    }
  }
}

/////////////////////////////////////////////////
int main(int _argc, char **_argv)
{
  // Load gazebo and run the transport system
  gazebo::client::setup(_argc, _argv);
  // Create a node, which provids functions to create publishers and subscribers
  gazebo::transport::NodePtr node;
  node = gazebo::transport::NodePtr(new gazebo::transport::Node());
  //node->Init();//equal to node.Init()
  node->Init();
  //Create a subscriber on the ''world_stats'' topic.
  //Gazebo publishes a stream of stats on this topic.
  gazebo::transport::SubscriberPtr sub = node->Subscribe("~/pose/info", cb);
  //world_stats is the proto filename, what does cb mean?
  //cb is a callback function

  //sub if the subscriber of node.
  // Busy wait loop...replace with your own code as needed.
  while (true)
    {
    //  std::cout<<"loop entered"<<std::endl;
      gazebo::common::Time::MSleep(10);
    }


  // Make sure to shut everything down.
  gazebo::client::shutdown();
}
