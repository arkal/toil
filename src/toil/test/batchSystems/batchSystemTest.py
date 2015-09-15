# Copyright (C) 2015 UCSC Computational Genomics Lab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from abc import ABCMeta, abstractmethod
import logging
import os
import time
import multiprocessing

from toil.common import Config
from toil.batchSystems.mesos.test import MesosTestSupport
from toil.batchSystems.parasolTestSupport import ParasolTestSupport
from toil.batchSystems.parasol import ParasolBatchSystem
from toil.batchSystems.singleMachine import SingleMachineBatchSystem
from toil.batchSystems.abstractBatchSystem import InsufficientSystemResources
from toil.test import ToilTest, needs_mesos, needs_parasol, needs_gridengine
import unittest

log = logging.getLogger(__name__)

# How many cores should be utilized by this test. The test will fail if the running system doesn't have at least that
# many cores.
#
numCores = 2

# How many jobs to run. This is only read by tests that run multiple jobs.
#
numJobs = 2

# How many cores to allocate for a particular job
#
numCoresPerJob = (numCores) / numJobs

memoryForJobs = 100e6

diskForJobs = 1000

testWorkerPath = os.path.join(os.path.dirname(__file__), "batchSystemTestWorker.sh")


class hidden:
    """
    Hide abstract base class from unittest's test case loader

    http://stackoverflow.com/questions/1323455/python-unit-test-with-base-and-sub-class#answer-25695512
    """

    class AbstractBatchSystemTest(ToilTest):
        """
        A base test case with generic tests that every batch system should pass
        """
        __metaclass__ = ABCMeta

        @abstractmethod
        def createBatchSystem(self):
            """
            :rtype: AbstractBatchSystem
            """
            raise NotImplementedError

        def _createDummyConfig(self):
            return Config()

        def setUp(self):
            super(hidden.AbstractBatchSystemTest, self).setUp()
            self.config = self._createDummyConfig()
            self.batchSystem = self.createBatchSystem()
            self.tempDir = self._createTempDir('testFiles')

        def tearDown(self):
            self.batchSystem.shutdown()
            super(hidden.AbstractBatchSystemTest, self).tearDown()

        def testAvailableCores(self):
            self.assertTrue(multiprocessing.cpu_count() >= numCores)
            
        def testRunJobs(self):
            jobCommand1 = "%s test1.txt" % testWorkerPath
            jobCommand2 = "%s test2.txt" % testWorkerPath
            job1 = self.batchSystem.issueBatchJob(jobCommand1, memory=memoryForJobs, cores=numCoresPerJob, disk=diskForJobs)
            job2 = self.batchSystem.issueBatchJob(jobCommand2, memory=memoryForJobs, cores=numCoresPerJob, disk=diskForJobs)

            issuedIDs = self.wait_for_jobs_to_issue(2)
            self.assertEqual(set(issuedIDs), set([job1, job2]))

            runningJobIDs = self.wait_for_jobs_to_start(2)
            self.assertEqual(set(runningJobIDs), set([job1, job2]))

            #Both jobs are running, so the test files should
            #be created almost instantly. Wait a small amount
            #of time to be absolutely sure.
            time.sleep(0.1)
            
            #Killing the jobs instead of allowing them to complete means this
            #test can run very quickly if the batch system issues and starts
            #the jobs quickly.
            self.batchSystem.killBatchJobs([job1, job2])
            self.assertEqual({}, self.batchSystem.getRunningBatchJobIDs())
            
            updatedJobIDs = []
            delay = 1000
            for i in range(2):
                updatedID, exitStatus = self.batchSystem.getUpdatedBatchJob(delay)
                self.assertEqual(exitStatus, 0)
                updatedJobIDs.append(updatedID)
            
            self.assertEqual(set(updatedJobIDs), set([job1, job2]))
                
            self.assertTrue(os.path.exists("test1.txt"))
            self.assertTrue(os.path.exists("test2.txt"))

        def testCheckResourceRequest(self):
            self.assertRaises(InsufficientSystemResources, self.batchSystem.checkResourceRequest,
                              memory=1000, cores=200, disk=1e9)
            self.assertRaises(InsufficientSystemResources, self.batchSystem.checkResourceRequest,
                              memory=5, cores=200, disk=1e9)
            self.assertRaises(InsufficientSystemResources, self.batchSystem.checkResourceRequest,
                              memory=1001e9, cores=1, disk=1e9)
            self.assertRaises(InsufficientSystemResources, self.batchSystem.checkResourceRequest,
                              memory=5, cores=1, disk=2e9)
            self.assertRaises(AssertionError, self.batchSystem.checkResourceRequest, memory=None,
                              cores=1, disk=1000)
            self.assertRaises(AssertionError, self.batchSystem.checkResourceRequest, memory=10,
                              cores=None, disk=1000)
            self.batchSystem.checkResourceRequest(memory=10, cores=1, disk=100)

        def testGetRescueJobFrequency(self):
            self.assertTrue(self.batchSystem.getRescueBatchJobFrequency() > 0)
            
        def wait_for_jobs_to_issue(self, numJobs):
            issuedIDs = []
            while not len(issuedIDs) == numJobs:
                issuedIDs = self.batchSystem.getIssuedBatchJobIDs()
            return issuedIDs
        
        def wait_for_jobs_to_start(self, numJobs):
            runningIDs = []
            while not len(runningIDs) == numJobs:
                time.sleep(0.1)
                runningIDs = self.batchSystem.getRunningBatchJobIDs().keys()
            return runningIDs

@needs_mesos
class MesosBatchSystemTest(hidden.AbstractBatchSystemTest, MesosTestSupport):
    """
    Tests against the Mesos batch system
    """

    def createBatchSystem(self):
        from toil.batchSystems.mesos.batchSystem import MesosBatchSystem
        self._startMesos(numCores)
        return MesosBatchSystem(config=self.config, maxCores=numCores, maxMemory=1e9, maxDisk=1001,
                                masterIP='127.0.0.1:5050')

    def tearDown(self):
        self._stopMesos()
        super(MesosBatchSystemTest, self).tearDown()


class SingleMachineBatchSystemTest(hidden.AbstractBatchSystemTest):
    def createBatchSystem(self):
        return SingleMachineBatchSystem(config=self.config, maxCores=numCores, maxMemory=1e9,
                                        maxDisk=1001)


@needs_parasol
class ParasolBatchSystemTest(hidden.AbstractBatchSystemTest, ParasolTestSupport):
    """
    Tests the Parasol batch system
    """
    def _createDummyConfig(self):
        config = super(ParasolBatchSystemTest, self)._createDummyConfig()
        # can't use _getTestJobStorePath since that method removes the directory
        config.jobStore = self._createTempDir('jobStore')
        return config

    def createBatchSystem(self):
        self._startParasol(numCores)
        return ParasolBatchSystem(config=self.config, maxCores=numCores, maxMemory=1e9,
                                  maxDisk=1001)

    def tearDown(self):
        self._stopParasol()
        super(ParasolBatchSystemTest, self).tearDown()

    @unittest.skip("Test fails because of issue https://github.com/BD2KGenomics/toil/issues/456.")
    def testRunJobs(self):
        pass

@needs_gridengine
class GridEngineTest(hidden.AbstractBatchSystemTest):
    """
    Tests against the GridEngine batch system
    """

    def _createDummyConfig(self):
        config = super(GridEngineTest, self)._createDummyConfig()
        # can't use _getTestJobStorePath since that method removes the directory
        config.jobStore = self._createTempDir('jobStore')
        return config

    def createBatchSystem(self):
        from toil.batchSystems.gridengine import GridengineBatchSystem
        return GridengineBatchSystem(config=self.config, maxCores=numCores, maxMemory=1000e9, maxDisk=1e9)

    @classmethod
    def setUpClass(cls):
        super(GridEngineTest, cls).setUpClass()
        logging.basicConfig(level=logging.DEBUG)
