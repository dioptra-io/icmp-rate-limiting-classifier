# Limited Ltd.

Limited Ltd. is a new tool for performing alias resolution, the task of grouping IP addresses into routers.
It is based on ICMP rate limiting.

The software is composed of two components, the prober (written in C++) and the classifier written in Python 3.

## Classifier

To get the classifier:

```
git clone https://gitlab.planet-lab.eu/cartography/icmp-rate-limiting-classifier.git
```
You have to first generate a classifier that works for your platform. We provide a default training set (resources/test_set_lr,0.05-0.10) but you can use your own if you want.
To generate the classifier, run:
```
python3 ICMPTrainModel.py
```
This will generate a file resources/random_forest_classifier4.joblib that is used by the script ICMPMain.py to classify the aliases.

## Prober
To get the prober:
```
git clone https://gitlab.planet-lab.eu/cartography/icmp-rate-limiting.git
```

### Prerequisites for the prober
The prober uses [libtins](http://libtins.github.io) to send and receive the packets.
Other dependencies are found in the CMakeLists.txt, noticeably you have to install [boost](https://www.boost.org). 
### Installing the prober
```
cd /prober/dir/
mkdir build
cd build
cmake ..
make
```
## Running Limited Ltd.
The prober is run by a python wrapper, and to launch it:
```
python3 ICMPMain.py /path/to/targets/ 
```
A lot of configurations options are available, see the file configuration file configuration/default.ini for more information.

<!--- ## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags).

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone who's code was used
* Inspiration
* etc
-->
