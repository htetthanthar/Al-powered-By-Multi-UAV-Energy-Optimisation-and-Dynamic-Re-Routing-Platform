pipeline {
  agent any

  options {
    disableConcurrentBuilds()
    timeout(time: 60, unit: 'MINUTES')
  }

  triggers {
    pollSCM('H/5 * * * *')
  }

  parameters {
    string(name: 'VPS_HOST', defaultValue: '', description: 'Public DNS name or IP address of the Debian VPS.')
    string(name: 'DEPLOY_PATH', defaultValue: '/opt/uav-live-ops', description: 'Remote directory used for this deployment.')
    booleanParam(name: 'PROVISION_GRAFANA', defaultValue: false, description: 'For a verified native Debian Grafana installation only; this restarts Grafana. Leave off for containerized Grafana.')
  }

  environment {
    COMPOSE_FILES = '-f Config/docker-compose.yml -f Config/docker-compose.vps.yml'
  }

  stages {
    stage('Validate') {
      steps {
        script {
          if (!(params.VPS_HOST ==~ /[A-Za-z0-9][A-Za-z0-9.-]*/)) {
            error('VPS_HOST must be a DNS name or IPv4 address.')
          }
          if (!(params.DEPLOY_PATH ==~ /\/opt\/[A-Za-z0-9_-]+/)) {
            error('DEPLOY_PATH must be a single application directory under /opt, such as /opt/uav-live-ops.')
          }
        }
        sh 'docker compose $COMPOSE_FILES config --quiet'
        sh "python3 -m compileall -q 'All sensor' models tests conftest.py"
      }
    }

    stage('Package') {
      steps {
        sh '''
          tar \
            --exclude=.git \
            --exclude=.venv \
            --exclude=__pycache__ \
            --exclude=.pytest_cache \
            --exclude=uav-live-ops.tgz \
            -czf uav-live-ops.tgz Config 'All sensor' models tests conftest.py requirements.txt index.html .dockerignore
        '''
      }
    }

    stage('Deploy to Debian VPS') {
      steps {
        script {
          if (!params.VPS_HOST?.trim()) {
            error('Set VPS_HOST before running the deployment.')
          }
        }
        withCredentials([
          sshUserPrivateKey(credentialsId: 'uav-vps-ssh', keyFileVariable: 'SSH_KEY', usernameVariable: 'SSH_USER'),
          file(credentialsId: 'uav-vps-known-hosts', variable: 'KNOWN_HOSTS')
        ]) {
          sh '''
            SSH_OPTIONS="-i $SSH_KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=$KNOWN_HOSTS"
            ssh $SSH_OPTIONS "$SSH_USER@${VPS_HOST}" "mkdir -p '${DEPLOY_PATH}'"
            scp $SSH_OPTIONS uav-live-ops.tgz "$SSH_USER@${VPS_HOST}:/tmp/uav-live-ops.tgz"
            ssh $SSH_OPTIONS "$SSH_USER@${VPS_HOST}" "tar -xzf /tmp/uav-live-ops.tgz -C '${DEPLOY_PATH}' && rm -f /tmp/uav-live-ops.tgz && cd '${DEPLOY_PATH}' && docker compose -p uav-live-ops -f Config/docker-compose.yml -f Config/docker-compose.vps.yml up -d --build telemetry-broadcaster prometheus"
          '''
        }
      }
    }

    stage('Provision existing Grafana') {
      when { expression { return params.PROVISION_GRAFANA } }
      steps {
        withCredentials([
          sshUserPrivateKey(credentialsId: 'uav-vps-ssh', keyFileVariable: 'SSH_KEY', usernameVariable: 'SSH_USER'),
          file(credentialsId: 'uav-vps-known-hosts', variable: 'KNOWN_HOSTS')
        ]) {
          sh '''
            SSH_OPTIONS="-i $SSH_KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=$KNOWN_HOSTS"
            ssh $SSH_OPTIONS "$SSH_USER@${VPS_HOST}" "if [ '$SSH_USER' = root ]; then bash '${DEPLOY_PATH}/Config/vps/install-grafana-provisioning.sh' '${DEPLOY_PATH}'; else sudo -n bash '${DEPLOY_PATH}/Config/vps/install-grafana-provisioning.sh' '${DEPLOY_PATH}'; fi"
          '''
        }
      }
    }

    stage('Verify live metrics') {
      steps {
        withCredentials([
          sshUserPrivateKey(credentialsId: 'uav-vps-ssh', keyFileVariable: 'SSH_KEY', usernameVariable: 'SSH_USER'),
          file(credentialsId: 'uav-vps-known-hosts', variable: 'KNOWN_HOSTS')
        ]) {
          sh '''
            SSH_OPTIONS="-i $SSH_KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=$KNOWN_HOSTS"
            ssh $SSH_OPTIONS "$SSH_USER@${VPS_HOST}" "python3 '${DEPLOY_PATH}/Config/vps/verify-live-metrics.py'"
          '''
        }
      }
    }
  }
}
