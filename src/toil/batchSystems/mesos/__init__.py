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
from collections import namedtuple

TaskData = namedtuple('TaskData', (
    # Time when the task was started
    'startTime',
    # Mesos' ID of the slave where task is being run
    'slaveID',
    # Mesos' ID of the executor running the task
    'executorID'))

ResourceRequirement = namedtuple('ResourceRequirement', (
    # Number of bytes (!) needed for a task
    'memory',
    # Number of CPU cores needed for a task
    'cores',
    # Number of bytes (!) needed for task on disk
    'disk'))

ToilJob = namedtuple('ToilJob', (
    # A job ID specific to this batch system implementation
    'jobID',
    # A ResourceRequirement tuple describing the resources needed by this job
    'resources',
    # The command to be run on the worker node
    'command',
    # The resource object representing the user script
    'userScript',
    # The resource object representing the toil source tarball
    'toilDistribution',
    # A dictionary with additional environment variables to be set on the worker process
    'environment',
    # A named tuple containing all the required info for cleaning up the worker node
    'workerCleanupInfo'))
